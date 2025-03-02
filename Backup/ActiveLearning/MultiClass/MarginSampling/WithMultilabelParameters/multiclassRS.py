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
len(os.listdir(image_dir))

# df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa045h/labelling/1_all_winding_images/")

df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
#train_df = pd.read_csv(df_dir + "/Splits_v2024-03-18/train_v2024-03-18_10%.csv")
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

list_data_frame = [train_df, test_df, val_df]                       # List of data frames
multiclass_labels = []

for x in range(len(list_data_frame)):                               # Iterating to the list of data frames
    labels = []
    for y in tqdm(range(list_data_frame[x].shape[0])):              # Iterating to all the images of selected data frame and assigning labels
        if list_data_frame[x]['multi-label_double_winding'][y] == 0:
        
            if list_data_frame[x]['multi-label_gap'][y] == 0:
                
                if list_data_frame[x]['multi-label_crossing'][y] == 0:
                    labels.append('0')
                else:
                    labels.append('1')

            else:
                if list_data_frame[x]['multi-label_crossing'][y] == 0:
                    labels.append('2')
                else:
                    labels.append('3')
        
        else:
            if list_data_frame[x]['multi-label_gap'][y] == 0:

                if list_data_frame[x]['multi-label_crossing'][y] == 0:
                    labels.append('4')
                else:
                    labels.append('5')

            else:
                if list_data_frame[x]['multi-label_crossing'][y] == 0:
                    labels.append('6')
                else:
                    labels.append('7')
    multiclass_labels.append(labels)   
   

print(len(multiclass_labels))

# MultiClass training data frame 
multiclass_train_df = train_df.assign(multiclass = multiclass_labels[0])
multiclass_train_df = multiclass_train_df[['image', 'multiclass']].dropna()

# MultiClass test data frame 
multiclass_test_df = test_df.assign(multiclass = multiclass_labels[1])
multiclass_test_df = multiclass_test_df[['image', 'multiclass']].dropna()

# MultiClass validation data frame 
multiclass_val_df = val_df.assign(multiclass = multiclass_labels[2])
multiclass_val_df = multiclass_val_df[['image', 'multiclass']].dropna()

print(f"multiclass_train_df shape:", multiclass_train_df.shape)
print(f"multiclass_test_df shape:", multiclass_test_df.shape)
print(f"multiclass_val_df shape:", multiclass_val_df.shape)

multiclass_train_df.head()

batch_size = 8


# Initialising Active learner and query strategy using modAL
from modAL.models import ActiveLearner
from modAL.uncertainty import margin_sampling

query_strategy = margin_sampling

# Function to find the nearest multiple of base for a given number
def batch(number, base):                                    # The function returns the nearest batch size multiple for the provided number of samples
    multiple = base * round(np.divide(number, base))        # Base refers to the batch size
    return multiple

power = 1
train_size = batch(int(np.ceil(np.power(10, power))), batch_size)                                                                       # To get samples in multiple of batch size
train_size

print(f"Calculated train_size: {train_size}")

# Shuffling the data frame 
multiclass_train_df = multiclass_train_df.sample(frac = 1, random_state = 1234)

# Intial set must contain sample from each class
initial_df = multiclass_train_df.groupby('multiclass').head(1).head(train_size)

# Assigning remaining samples as the pool of data
pool_df = multiclass_train_df.drop(initial_df.index)

initial_df.shape, pool_df.shape

initial_df

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.label_to_index = {label: idx for idx, label in enumerate(dataframe['multiclass'].unique())}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_str = self.dataframe.iloc[idx, 1]
        label = self.label_to_index[label_str]
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to prepare data
def prepare_data(dataframe, image_dir, transform):
    dataset = CustomDataset(dataframe=dataframe, image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images = []
    labels = []
    for imgs, lbls in tqdm(dataloader):
        images.append(imgs)
        labels.append(lbls)
    images = torch.cat(images)
    labels = torch.cat(labels)
    labels = F.one_hot(labels, num_classes=len(dataset.label_to_index)).float()
    return images, labels

# Prepare initial data
X_train_initial, y_train_initial = prepare_data(initial_df, image_dir, transform)

# Prepare pool data
X_pool, y_oracle = prepare_data(pool_df, image_dir, transform)

# Prepare validation data
X_val, y_val = prepare_data(multiclass_val_df, image_dir, transform)

# Prepare test data
X_test, y_test = prepare_data(multiclass_test_df, image_dir, transform)

print(f"X_train_initial shape: {X_train_initial.shape}, y_train_initial shape: {y_train_initial.shape}")
print(f"X_pool shape: {X_pool.shape}, y_oracle shape: {y_oracle.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

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
save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/MulticlassRS/{run_id}"
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

model = CustomDINONormModel(dino_model, num_classes=8).to(device)

#Hyperparameters
num_classes = 8
initial_samples = 8

from skorch.callbacks import EarlyStopping, Checkpoint, Callback

# Function to convert one-hot encoded labels to integer labels
def convert_one_hot_to_labels(y):
    return np.argmax(y, axis=1) if len(y.shape) > 1 else y

# Ensure that y_train_initial, y_pool_initial, y_val, and y_test are in the correct format
y_train_initial_np = convert_one_hot_to_labels(y_train_initial.clone().detach().cpu().numpy())
y_pool_initial_np = convert_one_hot_to_labels(y_oracle.clone().detach().cpu().numpy())
y_val_np = convert_one_hot_to_labels(y_val.clone().detach().cpu().numpy())
y_test_np = convert_one_hot_to_labels(y_test.clone().detach().cpu().numpy())

# Convert initial datasets to NumPy
X_train_initial_np = X_train_initial.clone().detach().cpu().numpy()
X_pool_np = X_pool.clone().detach().cpu().numpy()
X_val_np = X_val.clone().detach().cpu().numpy()
X_test_np = X_test.clone().detach().cpu().numpy()

print(f"X_train_initial_np shape: {X_train_initial_np.shape}, y_train_initial_np shape: {y_train_initial_np.shape}")
print(f"X_pool_np shape: {X_pool_np.shape}, y_pool_initial_np shape: {y_pool_initial_np.shape}")
print(f"X_val_np shape: {X_val_np.shape}, y_val_np shape: {y_val_np.shape}")
print(f"X_test_np shape: {X_test_np.shape}, y_test_np shape: {y_test_np.shape}")

# Initialize cumulative datasets
X_cumulative = X_train_initial_np.copy()
y_cumulative = y_train_initial_np.copy()

# Define scoring functions
f1_scorer = make_scorer(f1_score, average='micro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)

# Define a validation split
valid_ds = Dataset(X_val_np, y_val_np)
train_split = predefined_split(valid_ds)

# Early stopping
es = EarlyStopping(monitor='valid_loss', patience=15, lower_is_better=True)
cp = Checkpoint(dirname='model_checkpoints', monitor='valid_loss_best')




classifier = NeuralNetClassifier(
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
)

no_of_iterations = 14
best_f1_score = 0
wait = 0
patience = 13
#total_samples = len(initial_dataset)
total_samples = 8
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
X_train_initial_np = X_train_initial_np[:initial_samples]
y_train_initial_np = y_train_initial_np[:initial_samples]

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
            X_add, X_pool_np, y_add, y_pool_initial_np = train_test_split(X_pool_np, y_pool_initial_np, train_size=train_size)
            print(f"Number of samples in pool after split: {X_pool.shape}")
            print(f"Number of samples in X_add: {X_add.shape}")
            print(f"Number of samples in y_add: {y_add.shape}")
            print(f"Number of samples in y_oracle: {y_oracle.shape}")
            print(f"Number of samples in y_train_initial_np: {y_train_initial_np.shape}")
            print(f"Number of samples in y_pool_initial_np: {y_pool_initial_np.shape}")
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
            X_train_initial_np = np.concatenate((X_train_initial_np, X_pool_np), axis=0)
            y_train_initial_np = np.concatenate((y_train_initial_np, y_pool_initial_np), axis=0)
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





'''# Initialize the ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    query_strategy=margin_sampling,
    X_training=X_cumulative,
    y_training=y_cumulative
)'''



'''# Initialize EarlyStopping and other callbacks
total_samples = X_train_initial.shape[0]

# List to keep record of number of samples
no_of_samples = [X_train_initial.shape[0]]
performance_test_data = []
performance_val_data = []
acc_test_data = []
f1_test_data = []

# Function to get class indices from predictions
def get_class_indices(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return y_pred

# Initial performance calculation
print("Computing initial performance on test set...")
initial_y_pred = learner.predict(X_test.numpy())
initial_y_pred = get_class_indices(initial_y_pred)

# Convert y_test to class indices
y_test_class_indices = get_class_indices(y_test.numpy())

# Calculate initial F1 score and accuracy
initial_f1 = f1_score(y_test_class_indices, initial_y_pred, average='micro')
initial_acc = accuracy_score(y_test_class_indices, initial_y_pred)
print(f"Initial F1 score: {initial_f1}")
print(f"Initial accuracy: {initial_acc}")

import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Initialize tracking for Early Stopping
best_f1_score = 0.0
patience = 15  # Number of rounds to continue without improvement
wait = 0  # Current wait time

# Active learning loop parameters
n_queries = 13
acc_test_data = []
f1_test_data = []
power=1

for i in range(n_queries):
    # Determine the number of samples to query
    if i == 0:
        n_instances = 8
    else:
        power += 0.25
        n_instances = batch(int(np.ceil(np.power(10, power))), batch_size)
    print(f"\nQuery {i + 1}: Requesting {n_instances} samples.")
    print(f"Number of samples in pool before query: {X_pool_np.shape[0]}")

    # Perform the query
    query_idx, query_instance = learner.query(X_pool_np, n_instances=n_instances)
    X_query, y_query = X_pool_np[query_idx], y_pool_initial_np[query_idx]
    y_query = convert_one_hot_to_labels(y_query)

    # Update the cumulative datasets
    X_cumulative = np.vstack((X_cumulative, X_query)) if i > 0 else X_query
    y_cumulative = np.concatenate((y_cumulative, y_query)) if i > 0 else y_query

    # Retrain the learner with the cumulative data
    learner.teach(X=X_cumulative, y=y_cumulative, only_new=False)

    # Evaluate the learner's performance
    y_pred = learner.predict(X_test_np)
    accuracy = accuracy_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred, average='macro')
    acc_test_data.append(accuracy)
    f1_test_data.append(f1)

    # Output the performance metrics
    print(f"Accuracy after query {i + 1}: {accuracy}")
    print(f"F1 Score after query {i + 1}: {f1}")
    print(f"Number of samples used for retraining: {len(X_cumulative)}")

    # Early Stopping Check
    if f1 > best_f1_score:
        best_f1_score = f1
        wait = 0  # reset the wait counter
    else:
        wait += 1  # increment the wait counter
        if wait >= patience:
            print("Stopping early due to no improvement in F1 score.")
            break

    # Remove queried instances from the pool
    X_pool_np = np.delete(X_pool_np, query_idx, axis=0)
    y_pool_initial_np = np.delete(y_pool_initial_np, query_idx, axis=0)
    print(f"Number of samples in pool after query: {X_pool_np.shape[0]}")

# Log the final performance across all queries
print(f"Final F1 scores across iterations: {f1_test_data}")
print(f"Final accuracies across iterations: {acc_test_data}")

# Save the performance results
performance_filename = "/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/MSFixed/performance_resultsMS.npy"
np.save(performance_filename, {"f1_scores": f1_test_data, "accuracies": acc_test_data})
print(f"Performance results saved to {performance_filename}")
'''