import os
import modAL
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image 
from plotly import graph_objects, subplots
import torchvision
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.nn.functional import one_hot
import torch
import torch.nn as nn
import sys
from pathlib import Path
# Scorer function and training setup imports
from skorch.callbacks import EpochScoring
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch.classifier import NeuralNetClassifier
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

torch.multiprocessing.set_sharing_strategy('file_system')

image_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")

# Define the image directory path
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/multiclass")
#train_df = pd.read_csv(df_dir + "/Splits_v2024-03-18/train_v2024-03-18_10%.csv")
train_df = pd.read_csv(df_dir + "/newtrain.csv")
val_df = pd.read_csv(df_dir + "/newvalidation.csv")
test_df = pd.read_csv(df_dir + "/newtest.csv")

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

batch_size = 16
learning_rate = 0.000006
y_column = 'labels'

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

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

# Create the datasets
train_dataset = CustomImageDataset(train_df, image_dir, y_column, transform=train_transform)
val_dataset = CustomImageDataset(val_df, image_dir, y_column, transform=test_transform)
test_dataset = CustomImageDataset(test_df, image_dir, y_column, transform=test_transform)

'''# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

def extract_from_loader(loader):
    X_data = []
    y_data = []
    
    for inputs, labels in loader:
        X_data.append(inputs)
        y_data.append(labels)
    
    # Concatenate all the batches into a single tensor for both data and labels
    X_data = torch.cat(X_data, dim=0)
    y_data = torch.cat(y_data, dim=0)
    
    return X_data, y_data

# Extract training data
X_train, y_train = extract_from_loader(train_loader)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Extract validation data
X_val, y_val = extract_from_loader(val_loader)
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Extract test data
X_test, y_test = extract_from_loader(test_loader)
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


print(f"X_train_initial shape: {X_train.shape}, y_train_initial shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")'''

import csv
import os
import time
from skorch.callbacks import Callback

class CSVLogger(Callback):
    """Log epoch data to a CSV file."""
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        self.file_exist = os.path.exists(filename)  # Check if file already exists

    def on_epoch_end(self, net, **kwargs):
        logs = {key: net.history[-1, key] for key in self.fieldnames}

        if not self.file_exist:
            # Write headers to CSV
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            self.file_exist = True
        
        # Write data to CSV
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(logs)

# Initialize unique identifier for the current run
run_id = time.strftime("%Y%m%d-%H%M%S")

# Create directories for saving the results
save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/multiclass/TryMadi/{run_id}"
os.makedirs(save_dir, exist_ok=True)

# Field names for the logger
fieldnames = ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']

# Initialize CSVLogger with the path and fieldnames
csv_logger = CSVLogger(os.path.join(save_dir, "training_history.csv"), fieldnames)

import os

# Set proxy if necessary
os.environ['http_proxy'] = 'http://proxy:80'
os.environ['https_proxy'] = 'http://proxy:80'

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.callbacks import EpochScoring
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from modAL.models import ActiveLearner
from modAL.uncertainty import margin_sampling

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

# Initialize and configure the model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)

model = CustomDINONormModel(dino_model, num_classes=8).to(device)

#Hyperparameters
num_classes = 8
initial_samples = 8

from skorch.callbacks import EarlyStopping, Checkpoint, Callback

# Function to convert one-hot encoded labels to integer labels
def convert_one_hot_to_labels(y):
    return np.argmax(y, axis=1) if len(y.shape) > 1 else y

'''# Ensure that y_train_initial, y_pool_initial, y_val, and y_test are in the correct format
y_train_initial_np = convert_one_hot_to_labels(y_train.clone().detach().cpu().numpy())
#y_pool_initial_np = convert_one_hot_to_labels(y_oracle.clone().detach().cpu().numpy())
y_val_np = convert_one_hot_to_labels(y_val.clone().detach().cpu().numpy())
y_test_np = convert_one_hot_to_labels(y_test.clone().detach().cpu().numpy())'''

#y_train_np = y_train.clone().detach().cpu().numpy()
#y_val_np = y_val.clone().detach().cpu().numpy()
#y_test_np = y_test.clone().detach().cpu().numpy()

# Convert initial datasets to NumPy
#X_train_np = X_train.clone().detach().cpu().numpy()
#X_pool_np = X_pool.clone().detach().cpu().numpy()
#X_val_np = X_val.clone().detach().cpu().numpy()
#X_test_np = X_test.clone().detach().cpu().numpy()

#print(f"X_train_initial_np shape: {X_train_np.shape}, y_train_initial_np shape: {y_train_np.shape}")
#print(f"X_pool_np shape: {X_pool_np.shape}, y_pool_initial_np shape: {y_pool_initial_np.shape}")
#print(f"X_val_np shape: {X_val_np.shape}, y_val_np shape: {y_val_np.shape}")
#print(f"X_test_np shape: {X_test_np.shape}, y_test_np shape: {y_test_np.shape}")

'''# Initialize cumulative datasets
X_cumulative = X_train_initial_np.copy()
y_cumulative = y_train_initial_np.copy()'''

# Define scoring functions
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)

# Define a validation split
'''valid_ds = Dataset(X_val_np, y_val_np)
train_split = predefined_split(valid_ds)'''

train_split = predefined_split(val_dataset)

# Early stopping
es = EarlyStopping(monitor='valid_loss', patience=15, lower_is_better=True)
cp = Checkpoint(dirname='model_checkpoints', monitor='valid_loss_best')




'''classifier = NeuralNetClassifier(
    module=model,
    criterion=nn.CrossEntropyLoss(),
    #optimizer=optim.RMSprop,
    optimizer = optim.Adam,
    lr=0.00000075,
    max_epochs=100,
    train_split=predefined_split(valid_ds),  # Use predefined split for validation
    device=device,
    callbacks=[train_f1, valid_f1, es, cp, csv_logger],
    verbose=1
)'''
classes = train_df[y_column].unique()

print("Model initialized")
classifier = NeuralNetClassifier(
    module=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    lr=learning_rate,
    batch_size=batch_size,
    max_epochs=50,
    train_split=train_split,
    device=device,
    callbacks=[train_f1, valid_f1, es, cp, csv_logger],
    verbose=1,
    classes=classes
)

all_scores = []

'''for i in range(3):
    # Train the model using the training dataset
    classifier.fit(train_dataset)

    # Predict on the test dataset
    y_pred_test = classifier.predict(test_dataset)

    # Convert test labels to NumPy
    y_test_np = torch.tensor([label for _, label in test_dataset]).cpu().numpy()

    # Calculate test F1 score
    test_f1 = f1_score(y_test_np, y_pred_test, average='macro')  # Test labels should be in the same format
    print(f'Test F1-score for iteration {i+1}: {test_f1:.4f}')

    # Append test F1 score to the list
    all_scores.append(test_f1)'''

'''for i in range(3):
    # Train the model using the training dataset
    print(f"Training model on iteration {i+1}")
    classifier.fit(X=train_dataset, y=None)

    # Predict on the test dataset
    y_pred_test = classifier.predict(test_dataset)

    # Convert test labels to NumPy
    y_test_np = torch.tensor([label for _, label in test_dataset]).cpu().numpy()

    # Calculate test F1 score
    test_f1 = f1_score(y_test_np, y_pred_test, average='macro')  # Test labels should be in the same format
    print(f'Test F1-score for iteration {i+1}: {test_f1:.4f}')

    # Append test F1 score to the list
    all_scores.append(test_f1)'''

# Train the model using the training dataset
print("Training model on")
classifier.fit(X=train_dataset, y=None)

# Predict on the test dataset
y_pred_test = classifier.predict(test_dataset)

# Convert test labels to NumPy
y_test_np = torch.tensor([label for _, label in test_dataset]).cpu().numpy()

# Calculate test F1 score
test_f1 = f1_score(y_test_np, y_pred_test, average='macro')  # Test labels should be in the same format
print('Test F1-score for iteration: {:.4f}'.format(test_f1))

# Append test F1 score to the list
all_scores.append(test_f1)

print(f'Average test F1-score: {np.mean(all_scores):.4f}')