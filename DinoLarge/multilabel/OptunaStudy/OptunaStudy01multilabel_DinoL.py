import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import f1_score
from optuna.exceptions import TrialPruned
from pathlib import Path
import sys

# Paths and directories
image_dir = "/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/"
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
#train_df = pd.read_csv(df_dir + "/Splits_v2024-03-18/train_v2024-03-18_10%.csv")
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")
num_classes = 3
batch_size = 8

y_columns = train_df.drop(columns=["image", "binary_NOK"]).columns

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, y_columns, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.y_columns = y_columns
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('L').convert('RGB')
        labels = torch.tensor(self.dataframe.iloc[idx][self.y_columns].to_numpy().astype('float32'))
        if self.transform:
            image = self.transform(image)
        return image, labels
    
def print_hyperparameters(layer_freeze, dropout, learning_rate, optimizer_name, momentum_term, patience):
    print("Hyperparameters:")
    print(f"Layer freeze: {layer_freeze}")
    print(f"Dropout: {dropout}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Momentum term: {momentum_term}")
    print(f"Patience: {patience}")

class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, dropout):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.dino_model(x)
        x = self.classifier(x)
        return x
    
def define_model(layer_freeze, dropout):
    
    #local_model_path = '/home/hpc/iwfa/iwfa054h/.cache/torch/hub/facebookresearch_dinov2_main'
    dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    for param in list(dinov2_vitl14.parameters())[:layer_freeze]:
        param.requires_grad = False
    #dino_model = torch.hub.load(local_model_path, 'dinov2_vitl14', source='local')
    model = CustomDINONormModel(dinov2_vitl14, dropout)
    
    return model

# Optuna study setup and optimization
study = optuna.create_study(
    study_name='OptunaStudy01multilabel_DinoL',
    direction='maximize',
    storage='sqlite:///OptunaStudy01multilabel_DinoL.db',
    load_if_exists=True
)


def objective(trial):
    
    # Hyperparameters
    layer_freeze = trial.suggest_int('layer_freeze', 0, 200)
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
                                                                0.0025, 0.00025, 0.000025, 0.0000025, 0.00000025,
                                                                0.005, 0.0005, 0.00005, 0.000005, 0.0000005,
                                                                0.0075, 0.00075, 0.000075, 0.0000075, 0.00000075,])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    momentum_term = 0 if optimizer_name != 'SGD' else trial.suggest_float('momentum_term', 0.8, 0.99)
    patience = trial.suggest_int('patience', 3, 20)


    print("====================",f"Training of trial number:{trial.number}","====================")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print_hyperparameters(layer_freeze, dropout, learning_rate, optimizer_name, momentum_term, patience)

    train_dataset = CustomImageDataset(train_df, image_dir, y_columns, transform=transform)
    val_dataset = CustomImageDataset(val_df, image_dir, y_columns, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = define_model(layer_freeze, dropout)
    model.to(device)
    optimizer = None
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum_term)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_term)

    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 100

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        best_val_f1 = study.best_trial.value
    else:
        best_val_f1 = 0

    print("Best F1-score on Validation data until now:", best_val_f1)
    epochs_no_improve = 0
    trail_best = 0

    for epoch in range(num_epochs):
        model.train()
        all_labels = []
        all_predictions = []
        running_loss = 0.0
        
        for inputs, labels in (train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # Adjusted for Inception v3 without aux_logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs)).detach()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        model.eval()
        all_preds, all_labels = [], []
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in (val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training F1-score: {train_f1:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation F1-score: {val_f1:.4f}')

        trial.report(val_f1, epoch)
        if trial.should_prune():
            print("Best F1-score till now on Validation data:", best_val_f1)
            raise TrialPruned()

        if val_f1 < 0.1:
            print("Very bad trail")
            break

        if val_f1 > trail_best:
            trail_best = val_f1
            epochs_no_improve = 0
            print("Best F1-score till now on the current trail on Validation data:", trail_best)
            if(val_f1 > best_val_f1):
                best_val_f1 = val_f1
                print("Best F1-score till now on Validation data:", best_val_f1)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                print("Best F1-score till now on Validation data:", best_val_f1)
                break
    
    return best_val_f1



completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
if(len(completed_trials) > 0):
    best_trial = study.best_trial
    print("Best trial's number: ", best_trial.number)
    print(f"Best score: {best_trial.value}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")
        
        
        
total_trails_to_run = 200
completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
num_trials_completed = len(completed_trials)
pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
num_pruned_trials = len(pruned_trials)
print(f"Number of trials completed: {num_trials_completed}")
print(f"Number of pruned trials: {num_pruned_trials}")
print(f"Total number of trails completed: {num_trials_completed + num_pruned_trials}")
trials_to_run = max(0, total_trails_to_run - (num_trials_completed + num_pruned_trials))
print(f"Number of trials to run: {trials_to_run}")

study.optimize(objective, trials_to_run)

best_trial = study.best_trial
print("Best trial's number: ", best_trial.number)
print(f"Best score: {best_trial.value}")
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")