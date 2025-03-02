import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import optuna
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from optuna.exceptions import TrialPruned
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Define the image directory path
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
    def __init__(self, dino_model, num_classes=8):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
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

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

class_names = [str(i) for i in range(num_classes)]


# Initialize the list to store scores
all_scores = []

# Run multiple trials
for i in range(20):
    #local_model_path = '/home/hpc/iwfa/iwfa061h/.cache/torch/hub/facebookresearch_dinov2_main'
    #dino_model = torch.hub.load(local_model_path, 'dinov2_vitl14', source='local')
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    model = CustomDINONormModel(dino_model, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.000006)
    loss_function = nn.CrossEntropyLoss()
    
    epochs_no_improve = 0
    num_epochs = 30  # Increased number of epochs to give room for early stopping
    patience = 5  # Number of epochs to wait before stopping
    best_val_f1 = 0
    best_model_state = None

    # Training loop
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
            preds = torch.argmax(outputs, dim=1).detach()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_dataset)
        train_f1 = f1_score(all_labels, all_predictions, average='macro')

        # Validation loop
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
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training F1-score: {train_f1:.4f}, Validation Loss: {val_epoch_loss:.4f}, Validation F1-score: {val_f1:.4f}')

        # Check if validation F1-score improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()  # Save the best model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Testing loop
    model.eval()
    all_preds_test = []
    all_labels_test = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds_test.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())

    test_f1 = f1_score(all_labels_test, all_preds_test, average='macro')
    print(f'Test F1-score: {test_f1:.4f}')
    all_scores.append(test_f1)
    
    
    
cm = confusion_matrix(all_labels_test, all_preds_test)
print('Confusion Matrix')
print(cm)

    # Plot confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

    # Print classification report
print('Classification Report')
print(classification_report(all_labels_test, all_preds_test, target_names=class_names))

# Save the scores to a CSV file
pd.DataFrame(all_scores).to_csv("all_scores_DinoV2_5_trials_multiclass.csv")