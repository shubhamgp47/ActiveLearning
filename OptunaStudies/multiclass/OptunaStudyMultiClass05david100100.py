import torch
import optuna
import numpy as np
import torchvision.transforms as T
from PIL import Image
import os
#import cv2
#import json
#import glob
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.metrics import f1_score
from optuna.exceptions import TrialPruned
import random


# Paths and directories
image_dir = "/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/"
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/multiclass")
#train_df = pd.read_csv(df_dir + "/Splits_v2024-03-18/train_v2024-03-18_10%.csv")
train_df = pd.read_csv(df_dir + "/newtrain.csv")
val_df = pd.read_csv(df_dir + "/newvalidation.csv")
test_df = pd.read_csv(df_dir + "/newtest.csv")

# Specify the column containing class labels
y_column = 'labels'  # Update this if your label column has a different name

# Define the custom model
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, num_classes, dropout, fc_units, fc_units_2):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, fc_units),
            nn.LayerNorm(fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, fc_units_2),
            nn.Linear(fc_units_2, num_classes)
        )

    def forward(self, x):
        x = self.dino_model(x)
        x = self.classifier(x)
        return x

# Define the image transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, y_column, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.y_column = y_column
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        image = image.convert('RGB')
        label = torch.tensor(self.dataframe.iloc[idx][self.y_column]).long()

        if self.transform:
            image = self.transform(image)

        return image, label

# Get the number of classes
num_classes = 8

# Create the datasets
train_dataset = CustomImageDataset(train_df, image_dir, y_column, transform=train_transform)
val_dataset = CustomImageDataset(val_df, image_dir, y_column, transform=test_transform)
test_dataset = CustomImageDataset(test_df, image_dir, y_column, transform=test_transform)


num_classes = 8

def print_hyperparameters(batch_size, learning_rate, dropout, optimizer_name, momentum_term, patience, fc_units, fc_units_2):
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate:.10f}")
    print(f"Dropout: {dropout}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Momentum term: {momentum_term}")
    print(f"Patience: {patience}")
    print(f"Fully connected units: {fc_units}")
    print(f"Fully connected units: {fc_units_2}")

# Optuna objective function
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-10, 0.01, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8])
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    #num_frozen_params = trial.suggest_int('num_frozen_params', 0, 300)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    momentum_term = 0 if optimizer_name != 'SGD' else trial.suggest_float('momentum_term', 0.8, 0.99)
    patience = trial.suggest_int('patience', 3, 20)
    fc_units = trial.suggest_categorical('fc_units', [128, 256, 512, 1024, 2048])
    fc_units_2 = trial.suggest_categorical('fc_units_2', [128, 256, 512, 1024, 2048])
    
    #local_model_path = '/home/hpc/iwfa/iwfa061h/.cache/torch/hub/facebookresearch_dinov2_main'
    #dino_model = torch.hub.load(local_model_path, 'dinov2_vitl14', source='local')
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    model = CustomDINONormModel(dino_model, num_classes, dropout, fc_units, fc_units_2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum_term)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_term)
    criterion =nn.CrossEntropyLoss()

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=4)

    print_hyperparameters(batch_size, learning_rate, dropout, optimizer_name, momentum_term, patience, fc_units, fc_units_2)

    num_epochs = 30
    #patience = 3

    best_val_f1 = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        all_labels = []
        all_predictions = []
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, leave=True, desc=f'Epoch {epoch + 1}/{num_epochs} Training', ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).detach()
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
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training F1-score: {train_f1:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation F1-score: {val_f1:.4f}')

        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise TrialPruned()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

    return best_val_f1

# Define the study and optimize
'''storage_url = "sqlite:///windingdataoptuna.db"
study_name = "semilearn137_study"
study = optuna.create_study(study_name=study_name, storage=storage_url, direction="maximize", load_if_exists=True)'''

# Optuna study setup and optimization
study = optuna.create_study(
    study_name='optuna_study10multiclass05David',
    direction='maximize',
    storage='sqlite:///optuna_study10multiclass05David.db',
    load_if_exists=True
)

study.optimize(objective, n_trials=100)

# Output the best trial
best_trial = study.best_trial
print("Best trial's number: ", best_trial.number)
print(f"Best score: {best_trial.value}")
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

# Save the scores to a CSV file
all_scores = [trial.value for trial in study.trials if trial.value is not None]
pd.DataFrame(all_scores).to_csv("all_scores_DinoV2_optimized.csv")