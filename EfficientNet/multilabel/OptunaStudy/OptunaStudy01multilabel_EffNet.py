import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import os
from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import numpy as np
import optuna
from sklearn.metrics import f1_score
from optuna.exceptions import TrialPruned
import random
import multiprocessing


# Set working directory and paths
image_dir = os.path.abspath('C:/Users/localuserSKSG/Desktop/Datasets and images/linear_winding_images_with_labels/linear_winding_images_with_labels')
df_dir = os.path.abspath('C:/Users/localuserSKSG/Desktop/Datasets and images/datasets/')
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")

y_columns = train_df.drop(columns = ["image", "binary_NOK"]).columns

#Set CustomImageDataset
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, y_columns, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.y_columns = y_columns
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = f"{self.image_dir}/{self.dataframe.iloc[idx, 0]}"
        image = Image.open(img_name).convert('L')
        image = image.convert('RGB')
        labels = torch.tensor(self.dataframe.iloc[idx][self.y_columns].to_numpy().astype('float32'))

        if self.transform:
            image = self.transform(image)

        return image, labels
        
# print hyperparameters used in the current trail        
def print_hyperparameters(image_size, batch_size, learning_rate, fc_units, dropout_rate, layer_freeze_upto):
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate:.6f}")
    print(f"Fully connected layer: {fc_units}")
    print(f"Dropout rate: {dropout_rate:.6f}")
    print(f"Layer Freeze Upto: {layer_freeze_upto}")
    
    
# Define efficientnet_v2_l model     
def define_model(layer_freeze_upto, fc_units, dropout_rate, num_classes):
    
    model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

    cutoff_reached = False
    for name, param in model.named_parameters():
        if not cutoff_reached:
            if name == layer_freeze_upto:
                cutoff_reached = True
            param.requires_grad = False
        else:
            param.requires_grad = True
        
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, fc_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(fc_units, num_classes),
    )
    
    return model


import torch
from torch.utils.data import DataLoader
import optuna
from sklearn.metrics import f1_score

# Assuming you have defined your CustomImageDataset, print_hyperparameters, define_model, and other necessary components

def objective(trial):
    # Hyperparameters
    image_size = trial.suggest_categorical('image_size', [224])
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
                                                                0.0025, 0.00025, 0.000025, 0.0000025, 0.00000025,
                                                                0.005, 0.0005, 0.00005, 0.000005, 0.0000005,
                                                                0.0075, 0.00075, 0.000075, 0.0000075, 0.00000075])
    fc_units = trial.suggest_categorical('fc_units', [256, 512, 1024])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0, 0.5])
    layer_freeze_upto = trial.suggest_categorical('layer_freeze_upto', ['features.7.6.block.3.1.bias',
                                                                        'features.6.24.block.3.1.bias',
                                                                        'features.5.18.block.3.1.bias',
                                                                        'features.4.9.block.3.1.bias',
                                                                        'features.3.6.block.1.1.bias', 
                                                                        'features.2.6.block.1.1.bias',
                                                                        'features.1.3.block.0.1.bias',
                                                                        'features.0.1.bias'])
    num_classes = 3

    print("====================", f"Training of trial number: {trial.number}", "====================")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print_hyperparameters(image_size, batch_size, learning_rate, fc_units, dropout_rate, layer_freeze_upto)
    
    train_dataset = CustomImageDataset(train_df, image_dir, y_columns, transform=transform)
    val_dataset = CustomImageDataset(val_df, image_dir, y_columns, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = define_model(layer_freeze_upto, fc_units, dropout_rate, num_classes)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 100
    patience = 10

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best_val_f1 = study.best_trial.value if completed_trials else 0

    print("Best F1-score on Validation data until now:", best_val_f1)
    epochs_no_improve = 0
    trail_best = 0

    for epoch in range(num_epochs):
        model.train()
        all_labels = []
        all_predictions = []
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
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
            for inputs, labels in val_loader:
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
            raise optuna.TrialPruned()

        if val_f1 < 0.1:
            print("Very bad trial")
            break

        if val_f1 > trail_best:
            trail_best = val_f1
            epochs_no_improve = 0
            print("Best F1-score till now on the current trial on Validation data:", trail_best)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                print("Best F1-score till now on Validation data:", best_val_f1)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                print("Best F1-score till now on Validation data:", best_val_f1)
                break
    
    return best_val_f1

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    study_name = 'OptunaStudy01_multilabel_EffNet'
    storage_name = 'sqlite:///OptunaStudy01_multilabel_EffNet.db'  # Change this to your preferred storage

    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize', load_if_exists=True)
    
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
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
    print(f"Total number of trials completed: {num_trials_completed + num_pruned_trials}")
    trials_to_run = max(0, total_trails_to_run - (num_trials_completed + num_pruned_trials))
    print(f"Number of trials to run: {trials_to_run}")

    study.optimize(objective, n_trials=trials_to_run)

    best_trial = study.best_trial
    print("Best trial's number: ", best_trial.number)
    print(f"Best score: {best_trial.value}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"{key}: {value}")
