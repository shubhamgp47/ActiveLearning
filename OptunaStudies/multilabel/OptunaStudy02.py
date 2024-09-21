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
train_df = pd.read_csv('/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/train.csv')
val_df = pd.read_csv('/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/validation.csv')
test_df = pd.read_csv('/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/test.csv')
device = "cuda" if torch.cuda.is_available() else "cpu"

y_columns = train_df.drop(columns=["image"]).columns
num_epochs = 100

# Dataset class
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

# DINO model class
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, fc_units, dropout_rate):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(384, fc_units),
            nn.LayerNorm(fc_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units, len(y_columns)),
        )

    def forward(self, x):
        x = self.dino_model(x)
        x = self.classifier(x)
        return x

# Transform
def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Objective function for Optuna
def objective(trial):
    # Hyperparameters suggested by Optuna
    #image_size = trial.suggest_categorical('image_size', [128, 224, 256])
    batch_size = trial.suggest_categorical('batch_size', [8, 16])
    layer_freeze = trial.suggest_int('layer_freeze', 0, 175)
    #loss_name = trial.suggest_categorical('loss', ['binary_crossentropy', 'binary_focal_crossentropy'])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('learning_rate', 0.000001, 0.01, log=True)
    momentum_term = None if optimizer_name == 'Adam' else trial.suggest_float('momentum_term', 0.0, 0.9)
    fc_units = trial.suggest_int('FC_units', 64, 1024, step=64)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    #num_epochs = trial.suggest_int('num_epochs', 5, 200, step=5)
    patience = trial.suggest_int('patience', 3, 35)


    # Transform and DataLoader with suggested batch size
    transform = get_transform(224)
    train_dataset = CustomImageDataset(train_df, image_dir, y_columns, transform)
    val_dataset = CustomImageDataset(val_df, image_dir, y_columns, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    # Load DINO model
    dino_repo_path = Path('/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/dino-main02')
    sys.path.insert(0, str(dino_repo_path))
    from vision_transformer import vit_small
    dino_model = vit_small()
    model_state = torch.load(os.path.join(dino_repo_path, "dino_deitsmall16_pretrain02.pth"), map_location="cpu")
    dino_model.load_state_dict(model_state, strict=False)
    
    model = CustomDINONormModel(dino_model, fc_units, dropout_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Freeze layers
    for param in list(model.dino_model.parameters())[:layer_freeze]:
        param.requires_grad = False

    # Optimizer and loss function
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum_term)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum_term)

    '''if loss_name == 'binary_crossentropy':
        loss_function = nn.BCEWithLogitsLoss()
    elif loss_name == 'binary_focal_crossentropy':
        loss_function = nn.Loss'''

    loss_function = nn.BCEWithLogitsLoss()

    best_val_f1 = 0

    best_val_loss = float('inf')
    num_bad_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        all_labels = []
        all_predictions = []
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(torch.sigmoid(outputs)).detach()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(all_labels, all_predictions, average='macro')

        model.eval()
        all_preds = []
        all_labels = []
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.round(torch.sigmoid(outputs))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training F1-score: {train_f1:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation F1-score: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save the best model
            torch.save(model.state_dict(), 'best_model02.pth')

        # Prune the trial if it is not promising
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1

        if num_bad_epochs >= patience:
            break

    return best_val_f1

# Optuna study with SQLite storage
storage='sqlite:///optuna_study02.db'
study = optuna.create_study(study_name='optuna_study02', direction='maximize', storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=25)

# Print best trial
best_trial = study.best_trial
print(f'Best trial score: {best_trial.value}')
print(f'Best hyperparameters: {best_trial.params}')

# Load DINO model
dino_repo_path = Path('/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/dino-main02')
sys.path.insert(0, str(dino_repo_path))
from vision_transformer import vit_small

# Load the best model
best_model = CustomDINONormModel(vit_small(), best_trial.params['FC_units'], best_trial.params['dropout_rate'])
best_model.load_state_dict(torch.load('best_model02.pth'))
best_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Save all scores
test_dataset = CustomImageDataset(test_df, image_dir, y_columns, get_transform(224))
test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2)

all_scores = []

with torch.no_grad():
    all_preds_test = []
    all_labels_test = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs)
        preds = torch.round(torch.sigmoid(outputs))
        all_preds_test.extend(preds.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())

test_f1 = f1_score(all_labels_test, all_preds_test, average='macro')
print(f'Test F1-score: {test_f1:.4f}')
all_scores.append(test_f1)

pd.DataFrame(all_scores).to_csv("all_scores_DinoV2_XS_5_trials_learn_new_labels02.csv")