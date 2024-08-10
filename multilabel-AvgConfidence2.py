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
import numpy as np
import torch
from torch.utils.data import random_split, TensorDataset
from skorch.classifier import NeuralNetClassifier
import csv
import os
import time
from skorch.callbacks import Callback
from skorch.callbacks import EarlyStopping,Checkpoint
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch.optim as optim


# Directory and CSV setup
image_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")


import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
# Function to calculate adjusted batch size
def batch(number, base):
    return base * round(number / base)


def load_images(image_dir, df):
    images, labels = [], []
    for i in tqdm(range(df.shape[0])):
        image_path = os.path.join(image_dir, df.loc[i, "image"])
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        images.append(img)
        labels.append(df.drop(columns=['image', 'binary_NOK']).iloc[i].values.astype('float32'))
    return torch.stack(images), torch.tensor(labels)


X_train_initial, y_train_initial = load_images(image_dir, train_df)

X_val, y_val = load_images(image_dir, val_df)

X_test, y_test = load_images(image_dir, test_df)

# Function to create data loaders, handling GPU batching during training
def create_data_loader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Splitting datasets for initial training and pool
def split_datasets(X, y, train_size):
    return random_split(TensorDataset(X, y), [train_size, X.size(0) - train_size])


# Adjust variable names to avoid conflict
batch_size = 8  # Batch size
power = 1  # Dynamically adjust batch size for experimentation
train_size_value = int(np.ceil(np.power(10, power)))

# Assuming `batch_function` is the function you need to call instead of `batch`
train_size = train_size_value * batch_size

# Splitting the training dataset
initial_dataset, pool_dataset = split_datasets(X_train_initial, y_train_initial, train_size)

# Creating data loaders
# Creating data loaders
train_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)

pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)
val_loader = create_data_loader(X_val, y_val, batch_size)
test_loader = create_data_loader(X_test, y_test, batch_size)

print("Number of samples in unlabelled pool = ", len(pool_dataset))
print("Number of samples in training set = ", len(initial_dataset), "\n")

print("X_pool shape: ", pool_dataset[0][0].shape)  # Access a sample to print shape
print("y_oracle shape: ", pool_dataset[0][1].shape, "\n")

print("X_train_initial shape: ", initial_dataset[0][0].shape)
print("Y_train_initial shape: ", initial_dataset[0][1].shape)



# CSV Logger class
class CSVLogger(Callback):
    """Log epoch data to a CSV file."""
    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        self.file_exist = os.path.exists(filename)

    def on_epoch_end(self, net, **kwargs):
        logs = {key: net.history[-1, key] for key in self.fieldnames}

        if not self.file_exist:
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            self.file_exist = True
        
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(logs)


# Initialize unique identifier for the current run
run_id = time.strftime("%Y%m%d-%H%M%S")

# Create directories for saving the results
save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/MultiLabel/avg_confidence/final/{run_id}"
os.makedirs(save_dir, exist_ok=True)

# Field names for the logger
fieldnames = ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']

# Initialize CSVLogger with the path and fieldnames
csv_logger = CSVLogger(os.path.join(save_dir, "training_history.csv"), fieldnames)

# Custom model with specified modifications
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, num_classes):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.0032957439464482152),  # Dropout rate
            nn.Linear(1024, 512),  # Assuming last layer outputs 1024 features
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # Convert to (batch_size, num_channels, height, width)
        x = self.dino_model(x)
        x = self.classifier(x)
        return x

# Initialize and configure the model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
for param in list(dino_model.parameters())[:173]:  # Freeze first 173 layers
    param.requires_grad = False

model = CustomDINONormModel(dino_model, num_classes=3).to(device)
def tensors_to_numpy(loader):
    images, labels = [], []
    for data in loader:
        # Convert tensor to NumPy and then rearrange dimensions to match H x W x C
        batch_images = data[0].numpy()  # Shape of each: (batch_size, 3, 224, 224)
        batch_labels = data[1].numpy()  # Shape of each: (batch_size, 3)

        # Rearrange the dimensions to (batch_size, height, width, channels)
        batch_images = batch_images.transpose(0, 2, 3, 1)

        # Append all images and labels from the batch to lists
        images.append(batch_images)  # Append whole batch
        labels.append(batch_labels)  # Append whole batch

    # Concatenate all batches together
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    return images, labels

# Load and convert data
X_train_np, y_train_np = tensors_to_numpy(train_loader)
X_pool, y_pool = tensors_to_numpy(pool_loader)
X_test_np, y_test_np = tensors_to_numpy(test_loader)
X_val_np, y_val_np = tensors_to_numpy(val_loader)

# Check the shapes again after adjusting
print("Adjusted shapes after loading:")
print("X_train_np:", X_train_np.shape, "y_train_np:", y_train_np.shape)
print("X_pool:", X_pool.shape, "y_pool:", y_pool.shape)
print("X_test_np:", X_test_np.shape, "y_test_np:", y_test_np.shape)
print("X_val_np:", X_val_np.shape, "y_val_np:", y_val_np.shape)
# Skorch callbacks and classifier setup
# Skorch callbacks and classifier setup
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)
es = EarlyStopping(monitor='valid_loss', patience=10, lower_is_better=True)
cp = Checkpoint(dirname='model_checkpoints', monitor='valid_loss_best')

valid_ds = Dataset(X_val_np, y_val_np)
train_split = predefined_split(valid_ds)



valid_ds = Dataset(X_val, y_val)

classifier = NeuralNetClassifier(
    module=model,
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=optim.RMSprop,
    lr=0.00009732702526660113,
    max_epochs=50,
    train_split=predefined_split(valid_ds),  # Use predefined split for validation
    device=device,
    callbacks=[train_f1, valid_f1, EarlyStopping(patience=10), Checkpoint(dirname='model_checkpoints')],
    verbose=1
)

from modAL.models import ActiveLearner
from modAL.multilabel import min_confidence, avg_confidence

query_strategy = avg_confidence

initial_samples = 8

X_train_initial_np = X_train_np[:initial_samples]
y_train_initial_np = y_train_np[:initial_samples]

# Convert initial samples to the correct shape
X_cumulative = np.copy(X_train_initial_np)
y_cumulative = np.copy(y_train_initial_np)

print("X_cumulative shape: ", X_cumulative.shape)
print("y_cumulative shape: ", y_cumulative.shape)


# Active Learning setup
learner = ActiveLearner(
    estimator=classifier,
    query_strategy=avg_confidence,
    X_training=X_cumulative,
    y_training=y_cumulative
)

# Function to save the checkpoint
def save_checkpoint(iteration, X_data, y_data, filenames, save_dir):
    iteration_dir = os.path.join(save_dir, f"iteration_{iteration}")
    os.makedirs(iteration_dir, exist_ok=True)
    
    np.save(os.path.join(iteration_dir, "X_data.npy"), X_data)
    np.save(os.path.join(iteration_dir, "y_data.npy"), y_data)
    np.save(os.path.join(iteration_dir, "filenames.npy"), filenames)
    print(f"Checkpoint for iteration {iteration} saved.")

filenames_cumulative = [train_df['image'].iloc[i] for i in range(initial_samples)]

# Initialization of parameters for the active learning loop
n_queries = 14
batch_size = 8  # Ensure this is defined according to your use case
patience = 8
wait = 0
best_f1_score = 0.0
acc_test_data = []
f1_test_data = []
total_samples = 8  # Initial samples
n_instances_list = [8]  # Initial instance count

# Assuming X_pool, y_pool, X_test_np, y_test_np, and learner are already properly initialized and configured

# Initialize a list to keep track of filenames in the pool
filenames_pool = [image_file for image_file in train_df['image'].tolist()]

filenames_cumulative = []

# Start the active learning loop
for i in range(n_queries):
    if i == 0:
        n_instances = 8
    else:
        power += 0.25
        n_instances = batch(int(np.ceil(np.power(10, power))), batch_size)
    total_samples += n_instances
    n_instances_list.append(total_samples)
    
    print(f"\nQuery {i + 1}: Requesting {n_instances} samples.")
    print(f"Number of samples in pool before query: {X_pool.shape[0]}")

    

    with torch.device("cpu"):
        query_idx, _ = learner.query(X_pool, n_instances=n_instances)  # Assuming the learner is a modAL learner
        query_idx = np.unique(query_idx)
        query_idx = np.array(query_idx).flatten()  # Flatten in case the indices are nested or multidimensional

    # Extract the samples based on the query indices
    X_query = X_pool[query_idx]
    y_query = y_pool[query_idx]
    filenames_query = [filenames_pool[idx] for idx in query_idx]


    # Print the shapes to verify correctness
    print("Shape of X_query after indexing:", X_query.shape)

    if X_query.ndim != 4:
        raise ValueError(f"Unexpected number of dimensions in X_query: {X_query.ndim}")
    if X_query.shape[1:] != (224, 224, 3):
        raise ValueError(f"Unexpected shape in X_query dimensions: {X_query.shape}")

    # Concatenate the new queries to the cumulative dataset
    X_cumulative = np.vstack((X_cumulative, X_query))
    y_cumulative = np.vstack((y_cumulative, y_query))
    filenames_cumulative.extend(filenames_query)

    # Save checkpoint with filenames
    save_checkpoint(i + 1, X_cumulative, y_cumulative, filenames_cumulative, save_dir)

    # Retrain the learner with the cumulative data
    learner.teach(X=X_cumulative, y=y_cumulative)

    # Evaluate the learner's performance
    y_pred = learner.predict(X_test_np)
    accuracy = accuracy_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred, average='macro')
    acc_test_data.append(accuracy)
    f1_test_data.append(f1)

    print(f"Accuracy after query {i + 1}: {accuracy}")
    print(f"F1 Score after query {i + 1}: {f1}")


    # Early stopping check
    if f1 > best_f1_score:
        best_f1_score = f1
        wait = 0  # reset the wait counter
    else:
        wait += 1  # increment the wait counter
        if wait >= patience:
            print("Stopping early due to no improvement in F1 score.")
            break

    # Remove queried instances from the pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    filenames_pool = [filename for idx, filename in enumerate(filenames_pool) if idx not in query_idx]
    print(f"Number of samples in pool after query: {X_pool.shape[0]}")

# Log the final performance across all queries
print(f"Final F1 scores across iterations: {f1_test_data}")
print(f"Final accuracies across iterations: {acc_test_data}")