import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from skorch.callbacks import Callback, EarlyStopping, Checkpoint
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from modAL.models import ActiveLearner
from modAL.multilabel import avg_confidence, min_confidence
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import time
import csv
import plotly.graph_objects as go
import plotly.subplots as subplots
from sklearn.model_selection import train_test_split


# Directory and CSV setup
'''IMAGE_DIR = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")
TRAIN_CSV_PATH = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/Splits_v2024-03-18/train_v2024-03-18_10%.csv")
train_df = pd.read_csv(TRAIN_CSV_PATH)
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
TEST_CSV_PATH = os.path.join(df_dir, "test_v2024-03-18.csv")
test_df = pd.read_csv(TEST_CSV_PATH)
VAL_CSV_PATH = os.path.join(df_dir, "validation_v2024-03-18.csv")
val_df = pd.read_csv(VAL_CSV_PATH)'''

torch.multiprocessing.set_sharing_strategy('file_system')

# Directory and CSV setup
image_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")
#df_dir_25 = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/Splits_v2024-03-18/")
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")


def batch(number, base):
    return base * round(number / base)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, image_dir, df, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.df.loc[idx, "image"])
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        label = self.df.drop(columns=['image', 'binary_NOK']).iloc[idx].values.astype('float32')
        label = torch.tensor(label)

        return img, label


train_dataset = ImageDataset(image_dir, train_df, transform=transform)
val_dataset = ImageDataset(image_dir, val_df, transform=transform)
test_dataset = ImageDataset(image_dir, test_df, transform=transform)

# Initialize lists to store all images and labels
X_train_initial_list = []
y_train_initial_list = []

# Use DataLoader to iterate over the dataset in batches
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4)

# Extract and concatenate all images and labels
for images, labels in train_loader:
    X_train_initial_list.append(images)
    y_train_initial_list.append(labels)

# Combine all batches into single tensors
X_train_initial = torch.cat(X_train_initial_list, dim=0)
y_train_initial = torch.cat(y_train_initial_list, dim=0)

print("X_train_initial shape:", X_train_initial.shape)
print("y_train_initial shape:", y_train_initial.shape)

def split_datasets(X, y, train_size):
    return random_split(TensorDataset(X, y), [train_size, X.size(0) - train_size])


batch_size = 4
power = 1
train_size = int(np.ceil(np.power(10, power)))

initial_dataset, pool_dataset = split_datasets(X_train_initial, y_train_initial, train_size)

train_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/MultiLabel/random_samplingPreFinalBatchFix/{run_id}"
os.makedirs(save_dir, exist_ok=True)

# Field names for the logger
fieldnames = ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']
csv_logger = CSVLogger(os.path.join(save_dir, "training_history.csv"), fieldnames)

# Custom model class
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, num_classes):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.0032957439464482152),
            nn.Linear(1024, 512),
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
for param in list(dino_model.parameters())[:173]:
    param.requires_grad = False

model = CustomDINONormModel(dino_model, num_classes=3).to(device)

def tensors_to_numpy(loader):
    images, labels = [], []
    for data in loader:
        batch_images = data[0].numpy()
        batch_labels = data[1].numpy()
        batch_images = batch_images.transpose(0, 2, 3, 1)
        images.append(batch_images)
        labels.append(batch_labels)
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels

X_train_np, y_train_np = tensors_to_numpy(train_loader)
X_pool, y_pool = tensors_to_numpy(pool_loader)
X_test_np, y_test_np = tensors_to_numpy(test_loader)
X_val_np, y_val_np = tensors_to_numpy(val_loader)

# Check shapes
print("Adjusted shapes after loading:")
print("X_train_np:", X_train_np.shape, "y_train_np:", y_train_np.shape)
print("X_pool:", X_pool.shape, "y_pool:", y_pool.shape)
print("X_test_np:", X_test_np.shape, "y_test_np:", y_test_np.shape)
print("X_val_np:", X_val_np.shape, "y_val_np:", y_val_np.shape)

# Skorch callbacks and classifier setup
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)
es = EarlyStopping(monitor='valid_loss', patience=13, lower_is_better=True)
model_checkpoint = Checkpoint(dirname=os.path.join(save_dir, 'model_checkpoints'), monitor='valid_loss_best')

valid_ds = Dataset(X_val_np, y_val_np)
train_split = predefined_split(valid_ds)

classifier = NeuralNetClassifier(
    module=model,
    criterion=nn.BCEWithLogitsLoss(),
    #optimizer=optim.RMSprop,
    optimizer = optim.Adam,
    lr=0.00000075,
    max_epochs=100,
    train_split=predefined_split(valid_ds),  # Use predefined split for validation
    device=device,
    callbacks=[train_f1, valid_f1, es, model_checkpoint, csv_logger],
    verbose=1
)

no_of_iterations = 14
best_f1_score = 0
wait = 0
patience = 13
total_samples = len(initial_dataset)
acc_test_data = []
f1_test_data = []

# File to save selected sample names per iteration
samples_log_file = os.path.join(save_dir, "sample_selection_log.csv")
with open(samples_log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Sample Names"])

'''for i in range(no_of_iterations):
    print(f"\nIteration: {i + 1}")
    print(f"Number of samples in training set: {len(X_train_np)}")
    
    classifier.fit(X_train_np, y_train_np)
    
    y_pred = classifier.predict(X_test_np)
    
    test_f1 = f1_score(y_test_np, y_pred, average='macro')
    test_acc = accuracy_score(y_test_np, y_pred)
    
    acc_test_data.append(test_acc)
    f1_test_data.append(test_f1)
    
    print(f"Test F1 Score: {test_f1}")
    print(f"Test Accuracy: {test_acc}")
    
    if i < (no_of_iterations - 1):
        power += 0.25
        train_size = batch(int(np.ceil(np.power(10, power))), batch_size)
        
        if train_size <= X_pool.shape[0]:
            X_add, X_pool, y_add, y_pool = train_test_split(X_pool, y_pool, train_size=train_size)
        else:
            raise OverflowError(f"{train_size} informative samples are not available in the unlabeled pool.")
        
        X_train_np = np.concatenate((X_train_np, X_add), axis=0)
        y_train_np = np.concatenate((y_train_np, y_add), axis=0)
        
        print(f"Number of samples added in this iteration: {train_size}")
        
        selected_sample_names = train_df.loc[:train_size - 1, "image"].tolist()
        with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1] + selected_sample_names)
    else:
        print("\nRandom sampling loop ended successfully!")
    
    torch.cuda.empty_cache()'''

initial_samples = 8
X_train_initial_np = X_train_np[:initial_samples]
y_train_initial_np = y_train_np[:initial_samples]

for i in range(no_of_iterations):
    # Determine the number of instances to add in the current iteration
    if i == no_of_iterations - 1:  # Last iteration
        n_instances = X_pool.shape[0]
    else:
        n_instances = min(batch(int(np.ceil(np.power(10, power))), batch_size), X_pool.shape[0])

    print(f"\nIteration: {i + 1}")
    print(f"Number of samples in training set: {len(X_train_initial_np)}")
    
    # Train the classifier with the current training set
    classifier.fit(X_train_initial_np, y_train_initial_np)
    
    # Evaluate on the test set
    y_pred = classifier.predict(X_test_np)
    test_f1 = f1_score(y_test_np, y_pred, average='macro')
    test_acc = accuracy_score(y_test_np, y_pred)
    
    acc_test_data.append(test_acc)
    f1_test_data.append(test_f1)
    
    print(f"Test F1 Score: {test_f1}")
    print(f"Test Accuracy: {test_acc}")
    
    # If there are more iterations, add new samples to the training set
    if i < no_of_iterations - 1:
        power += 0.25
        # Determine the train size and ensure it does not exceed the remaining pool size
        train_size = min(n_instances, X_pool.shape[0])
        
        if train_size > 0:
            # Split the pool to get new training instances
            X_add, X_pool, y_add, y_pool = train_test_split(X_pool, y_pool, train_size=train_size)
        
            # Add the new samples to the training set
            X_train_initial_np = np.concatenate((X_train_initial_np, X_add), axis=0)
            y_train_initial_np = np.concatenate((y_train_initial_np, y_add), axis=0)
        
            print(f"Number of samples added in this iteration: {train_size}")
        
            # Log the names of selected samples
            selected_sample_names = train_df.loc[:train_size - 1, "image"].tolist()
            with open(samples_log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i + 1] + selected_sample_names)
        else:
            print("No more samples to add in this iteration.")
    else:
        # Final iteration: Add all remaining samples from the pool to the training set
        print("Welcome to the final iteration my friend!")
        if X_pool.shape[0] > 0:
            X_train_initial_np = np.concatenate((X_train_initial_np, X_pool), axis=0)
            y_train_initial_np = np.concatenate((y_train_initial_np, y_pool), axis=0)
            print(f"Final iteration: Added the remaining {X_pool.shape[0]} samples to the training set.")
            X_pool, y_pool = np.array([]), np.array([])  # Empty the pool after adding the samples
        
        print("\nRandom sampling loop ended successfully!")


    
    '''# If there are more iterations, add new samples to the training set
    if i < (no_of_iterations - 1):
        power += 0.25
        
        # Determine the train size and ensure it does not exceed the remaining pool size
        train_size = min(batch(int(np.ceil(np.power(10, power))), batch_size), X_pool.shape[0])
        
        # Split the pool to get new training instances
        X_add, X_pool, y_add, y_pool = train_test_split(X_pool, y_pool, train_size=train_size)
        
        # Add the new samples to the training set
        X_train_initial_np = np.concatenate((X_train_initial_np, X_add), axis=0)
        y_train_initial_np = np.concatenate((y_train_initial_np, y_add), axis=0)
        
        print(f"Number of samples added in this iteration: {train_size}")
        
        # Log the names of selected samples
        selected_sample_names = train_df.loc[:train_size - 1, "image"].tolist()
        with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1] + selected_sample_names)
    else:
        # Final iteration, if any remaining samples are there, add them to the training set
        if X_pool.shape[0] > 0:
            X_train_initial_np = np.concatenate((X_train_initial_np, X_pool), axis=0)
            y_train_initial_np = np.concatenate((y_train_initial_np, y_pool), axis=0)
            print(f"Final iteration: Added the remaining {X_pool.shape[0]} samples to the training set.")
        
        print("\nRandom sampling loop ended successfully!")'''

    # Clear GPU cache
    torch.cuda.empty_cache()



performance_test_data = [acc_test_data, f1_test_data]

print('F1 Test Data:', f1_test_data)
print('Accuracy Test Data:', acc_test_data)
print('Performance Test Data:', performance_test_data)


# Save performance metrics
performance_metrics_file = os.path.join(save_dir, "performance_metrics.csv")
with open(performance_metrics_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Accuracy", "F1 Score"])
    for iteration, (acc, f1) in enumerate(zip(acc_test_data, f1_test_data), start=1):
        writer.writerow([iteration, acc, f1])