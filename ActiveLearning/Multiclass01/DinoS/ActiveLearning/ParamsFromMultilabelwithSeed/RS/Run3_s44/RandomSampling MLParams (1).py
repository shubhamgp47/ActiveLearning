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

#torch.multiprocessing.set_sharing_strategy('file_system')

seed_value = 43

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

image_dir = os.path.abspath('D:/linear_winding_images_with_labels/')
df_dir = os.path.abspath('D:/datasets/')
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")

'''image_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")
#df_dir_25 = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/Splits_v2024-03-18/")
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")'''

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

# Define the specific image names you want to include in the initial dataset
predefined_image_names = ["Spule035_Image0269.jpg", "Spule020_Image0191.jpg", "Spule030_Image0310.jpg", "Spule013_Image0317.jpg",
                          "Spule006_Image0201.jpg", "Spule020_Image0292.jpg", "Spule012_Image0854.jpg", "Spule020_Image0498.jpg"]


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

def batch(number, base):                                    # The function returns the nearest batch size multiple for the provided number of samples
    multiple = base * round(np.divide(number, base))        # Base refers to the batch size
    return multiple

power = 1
train_size = batch(int(np.ceil(np.power(10, power))), batch_size)                                                                       # To get samples in multiple of batch size
train_size

print(f"Calculated train_size: {train_size}")

# Shuffling the data frame 
multiclass_train_df = multiclass_train_df.sample(frac = 1, random_state = 1234)

# Filter multiclass_train_df to contain only the rows with these specific images
initial_df = multiclass_train_df[multiclass_train_df['image'].isin(predefined_image_names)]
initial_df = initial_df.set_index('image').loc[predefined_image_names].reset_index()

missing_images = [img for img in predefined_image_names if img not in multiclass_train_df['image'].values]
if missing_images:
    print("These images are missing from multiclass_train_df:", missing_images)
    assert not missing_images, "Initial DataFrame doesn't contain the specified image."

# Verify if initial_df contains only the predefined images and that it has the desired shape
assert initial_df['image'].tolist() == predefined_image_names, "Initial DataFrame doesn't contain the specified images."
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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = int(self.dataframe.iloc[idx, 1])  # Ensure label is an integer
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#if __name__ == "__main__":
initial_dataset = CustomDataset(initial_df, image_dir, transform)
train_dataset = CustomDataset(multiclass_train_df, image_dir, transform)
pool_dataset = CustomDataset(pool_df, image_dir, transform)
val_dataset = CustomDataset(multiclass_val_df, image_dir, transform)
test_dataset = CustomDataset(multiclass_test_df, image_dir, transform)

batch_size = 8

initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
#save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/multiclass/MulticlassRS/MulticlassRS/MyNewParameters/DinoSmall/Micro/{run_id}"
save_dir = os.path.abspath('D:/Shubham/results/Multiclass01/RandomSampling/ParamsFromMultilabel/Run2/')
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
            nn.Dropout(0.35659850739606247),
            nn.Linear(384, 256),      # The output size of DINO ViT-S is 384
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, num_classes)
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

dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
for param in list(dino_model.parameters())[:65]:
    param.requires_grad = False

model = CustomDINONormModel(dino_model, num_classes=8).to(device)

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

X_initial_np, y_initial_np = tensors_to_numpy(initial_loader)
X_train_np, y_train_np = tensors_to_numpy(train_loader)
X_pool_np, y_pool_np = tensors_to_numpy(pool_loader)
X_test_np, y_test_np = tensors_to_numpy(test_loader)
X_val_np, y_val_np = tensors_to_numpy(val_loader)

# Check shapes
print("Adjusted shapes after loading:")
print("X_initial_np:", X_initial_np.shape, "y_initial_np:", y_initial_np.shape)
print("X_train_np:", X_train_np.shape, "y_train_np:", y_train_np.shape)
print("X_pool_np:", X_pool_np.shape, "y_pool_np:", y_pool_np.shape)
print("X_test_np:", X_test_np.shape, "y_test_np:", y_test_np.shape)
print("X_val_np:", X_val_np.shape, "y_val_np:", y_val_np.shape)

#Hyperparameters
num_classes = 8
initial_samples = 8

from skorch.callbacks import EarlyStopping, Checkpoint, Callback
# Skorch callbacks and classifier setup
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)
es = EarlyStopping(monitor='valid_loss', patience=13, lower_is_better=True)
model_checkpoint = Checkpoint(dirname=os.path.join(save_dir, 'model_checkpoints'), monitor='valid_loss_best')

valid_ds = Dataset(X_val_np, y_val_np)
train_split = predefined_split(valid_ds)

'''classifier = NeuralNetClassifier(
    module=model,
    criterion=nn.CrossEntropyLoss,
    #criterion = nn.BCEWithLogitsLoss,
    optimizer=optim.RMSprop,
    lr=0.000002,
    max_epochs=100,
    train_split=train_split,
    device=device,
    callbacks=[train_f1, valid_f1, es, model_checkpoint, csv_logger],
    verbose=1
)'''

classifier=NeuralNetClassifier(
        module=model,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        optimizer__momentum=0.14729309193472406,
        lr=0.0001375803586556554,
        max_epochs=100,
        train_split=predefined_split(Dataset(X_val_np, y_val_np)),  # Validation set split
        device=device,
        callbacks=[train_f1, valid_f1, es, model_checkpoint, csv_logger],
        verbose=1
    )

no_of_iterations = 14
best_f1_score = 0
wait = 0
patience = 13
#total_samples = len(initial_dataset)
total_samples = 8
#acc_test_data = []
f1_test_data = []

# File to save selected sample names per iteration
samples_log_file = os.path.join(save_dir, "sample_selection_log.csv")
with open(samples_log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Query_Iteration", "Selected_Image"])

initial_samples = 8
X_train_initial_np = X_initial_np
y_train_initial_np = y_initial_np


# Add an index column to keep track of original indices in X_pool_np
pool_indices = np.arange(len(X_pool_np))
POWER = 1
#cumulative_sample_names = []
'''for i in range(no_of_iterations):
    if i == 0:
        n_instances = initial_samples  # Using existing initial samples
        print(f"\nIteration {i + 1}: Using the initial {n_instances} samples already in X_train_initial_np and y_train_initial_np.")
        print(f"Number of samples in training set: {len(X_train_initial_np)}")
        # Log the initially selected sample names (for the first iteration)
        initial_sample_names = initial_df['image'].tolist()  # Assuming 'image' column holds the image names
        initial_sample_names = initial_df.iloc[:,0].tolist()

        with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1] + initial_sample_names)
    elif i == no_of_iterations - 1:  # Last iteration
        n_instances = X_pool_np.shape[0]  # Select all remaining samples in the final iteration
        print(f"\nIteration {i + 1}: Requesting all remaining samples.")
    else:
        POWER += 0.25
        n_instances = batch(int(np.ceil(np.power(10, POWER))), batch_size)
        print(f"\nIteration {i + 1}: Requesting {n_instances} samples.")

    if i != no_of_iterations - 1:
        # Train the classifier with the current training set
        classifier.fit(X_train_initial_np, y_train_initial_np)

        # Evaluate on the test set
        y_pred = classifier.predict(X_test_np)
        test_f1 = f1_score(y_test_np, y_pred, average='micro')
        test_acc = accuracy_score(y_test_np, y_pred)

        acc_test_data.append(test_acc)
        f1_test_data.append(test_f1)

        print(f"Test F1 Score: {test_f1}")
        print(f"Test Accuracy: {test_acc}")
    # This was previous working logic
    # If there are more iterations, add new samples to the training set
    if i != 0 and i < no_of_iterations - 1:
        #power += 0.25
        # Determine the train size and ensure it does not exceed the remaining pool size
        #train_size = min(n_instances, X_pool_np.shape[0])
        train_size = min(n_instances, X_pool_np.shape[0])

        if train_size > 0:
            # Split the pool to get new training instances
            #X_add, X_pool_np, y_add, y_pool_np = train_test_split(X_pool_np, y_pool_np, train_size=train_size, random_state=47)
            X_add, X_pool_np, y_add, y_pool_np = train_test_split(X_pool_np, y_pool_np, train_size=train_size)
            print(f"Number of samples in pool after split: {X_pool_np.shape}")
            print(f"Number of samples in X_add: {X_add.shape}")
            print(f"Number of samples in y_add: {y_add.shape}")

            # Add the new samples to the training set
            X_train_initial_np = np.concatenate((X_train_initial_np, X_add), axis=0)
            y_train_initial_np = np.concatenate((y_train_initial_np, y_add), axis=0)
            print(f"Number of samples in training set: {len(X_train_initial_np)}")
            print(f"Number of samples added in this iteration: {train_size}")
    # New Logic
    # Check if not the first or last iteration
    if i != 0 and i < no_of_iterations - 1:
        train_size = min(n_instances, len(pool_indices))

        if train_size > 0:
            # Split the pool indices to get indices for new training instances
            selected_indices, remaining_indices = train_test_split(pool_indices, train_size=train_size)

            # Use the indices to extract the samples from X_pool_np and y_pool_np
            X_add = X_pool_np[selected_indices]
            y_add = y_pool_np[selected_indices]

            # Update pool_indices to only keep remaining samples for the next iteration
            pool_indices = remaining_indices

            # Add the selected samples to the training set
            X_train_initial_np = np.concatenate((X_train_initial_np, X_add), axis=0)
            y_train_initial_np = np.concatenate((y_train_initial_np, y_add), axis=0)

            # Log the names of selected samples
            sample_names = multiclass_train_df.iloc[selected_indices, 0].values.tolist()  # Assuming names in first column
            with open(samples_log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i + 1] + sample_names)
                # Log the names of selected samples
        else:
            print("No more samples to add in this iteration.")
    #previous working logic
    

    elif i == no_of_iterations - 1:
        # Final iteration: Select all remaining samples
        X_add = X_pool_np[pool_indices]
        y_add = y_pool_np[pool_indices]

        # Add final remaining samples to the training set
        X_train_initial_np = np.concatenate((X_train_initial_np, X_add), axis=0)
        y_train_initial_np = np.concatenate((y_train_initial_np, y_add), axis=0)

        # Log final sample names
        sample_names = multiclass_train_df.iloc[pool_indices, 0].values.tolist()
        with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1] + sample_names)

        # Clear the pool after the final iteration
        X_pool_np, y_pool_np = np.array([]), np.array([])

        # **Train and evaluate the classifier one final time after adding the remaining samples**
        print(f"Number of samples in training set (final iteration): {len(X_train_initial_np)}")
        classifier.fit(X_train_initial_np, y_train_initial_np)

        # Final evaluation on the test set
        y_pred = classifier.predict(X_test_np)
        final_test_f1 = f1_score(y_test_np, y_pred, average='micro')
        final_test_acc = accuracy_score(y_test_np, y_pred)

        acc_test_data.append(final_test_acc)
        f1_test_data.append(final_test_f1)

        print(f"Final Test F1 Score: {final_test_f1}")
        print(f"Final Test Accuracy: {final_test_acc}")

        print("\nRandom sampling loop ended successfully!")

    checkpoint_path = f"D:/Shubham/results/Multiclass01/RandomSampling/model_checkpoint_iteration_{i}.pt"
    torch.save(classifier.module_.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # Clear GPU cache
    torch.cuda.empty_cache()
'''
'''for i in range(no_of_iterations):
    if i == 0:
        n_instances = len(X_train_initial_np)  # Using existing initial samples
        print(f"\nIteration {i + 1}: Using the initial {n_instances} samples already in X_train_initial_np and y_train_initial_np.")
        print(f"Number of samples in training set: {len(X_train_initial_np)}")
        
        # Log the initially selected sample names (for the first iteration)
        initial_sample_names = initial_df['image'].tolist()
        with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1] + initial_sample_names)
        classifier.fit(X_train_initial_np, y_train_initial_np)

        # Evaluate on the test set
        y_pred = classifier.predict(X_test_np)
        test_f1 = f1_score(y_test_np, y_pred, average='micro')
        #test_acc = accuracy_score(y_test_np, y_pred)

        #acc_test_data.append(test_acc)
        f1_test_data.append(test_f1)
        print(f'In iteration : {i+1}')
        print(f"Test F1 Score: {test_f1}")
    else:
        # For iterations from the second onward, select new samples
        if i == no_of_iterations - 1:
            n_instances = X_pool_np.shape[0]  # Select all remaining samples in the final iteration
            print(f"\nIteration {i + 1}: Requesting all remaining samples.")
        else:
            POWER += 0.25
            n_instances = batch(int(np.ceil(np.power(10, POWER))), batch_size)
            print(f"\nIteration {i + 1}: Requesting {n_instances} samples.")
        
        # Only add new samples if not in the first iteration
        if i > 0 and i < no_of_iterations - 1:
            train_size = min(n_instances, len(pool_indices))

            if train_size > 0:
                # Split the pool indices to get indices for new training instances
                selected_indices, remaining_indices = train_test_split(pool_indices, train_size=train_size)

                # Use the indices to extract the samples from X_pool_np and y_pool_np
                X_add = X_pool_np[selected_indices]
                y_add = y_pool_np[selected_indices]

                # Update pool_indices to only keep remaining samples for the next iteration
                pool_indices = remaining_indices

                # Add the selected samples to the training set
                X_train_initial_np = np.concatenate((X_train_initial_np, X_add), axis=0)
                y_train_initial_np = np.concatenate((y_train_initial_np, y_add), axis=0)

                # Log the names of selected samples
                sample_names = multiclass_train_df.iloc[selected_indices, 0].values.tolist()  # Assuming names in first column
                with open(samples_log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i + 1] + sample_names)
            else:
                print("No more samples to add in this iteration.")

        # Train the classifier with the updated training set
        classifier.fit(X_train_initial_np, y_train_initial_np)

        # Evaluate on the test set
        y_pred = classifier.predict(X_test_np)
        test_f1 = f1_score(y_test_np, y_pred, average='micro')
        #test_acc = accuracy_score(y_test_np, y_pred)

        #acc_test_data.append(test_acc)
        f1_test_data.append(test_f1)
        print(f'In iteration : {i+1}')
        print(f"Test F1 Score: {test_f1}")
        #print(f"Test Accuracy: {test_acc}")

    # Final iteration: Select all remaining samples
    if i == no_of_iterations - 1:
        X_add = X_pool_np[pool_indices]
        y_add = y_pool_np[pool_indices]

        # Add final remaining samples to the training set
        X_train_initial_np = np.concatenate((X_train_initial_np, X_add), axis=0)
        y_train_initial_np = np.concatenate((y_train_initial_np, y_add), axis=0)

        # Log final sample names
        sample_names = multiclass_train_df.iloc[pool_indices, 0].values.tolist()
        with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1] + sample_names)

        # Clear the pool after the final iteration
        X_pool_np, y_pool_np = np.array([]), np.array([])

        # **Train and evaluate the classifier one final time after adding the remaining samples**
        print(f"Number of samples in training set (final iteration): {len(X_train_initial_np)}")
        classifier.fit(X_train_initial_np, y_train_initial_np)

        # Final evaluation on the test set
        y_pred = classifier.predict(X_test_np)
        final_test_f1 = f1_score(y_test_np, y_pred, average='micro')
        #final_test_acc = accuracy_score(y_test_np, y_pred)

        #acc_test_data.append(final_test_acc)
        f1_test_data.append(final_test_f1)

        print(f"Final Test F1 Score: {final_test_f1}")
        #print(f"Final Test Accuracy: {final_test_acc}")

    # Save model checkpoint after each iteration
    checkpoint_path = f"D:/Shubham/results/Multiclass01/RandomSampling/model_checkpoint_iteration_{i}.pt"
    #torch.save(classifier.module_.state_dict(), checkpoint_path)
    torch.save(classifier.module.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # Clear GPU cache
    torch.cuda.empty_cache()
'''

def train_and_evaluate(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='micro')
    return test_f1

def log_sample_names(iteration, sample_names, log_file):
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([iteration] + sample_names)

'''def select_new_samples(pool_indices, X_pool, y_pool, n_instances, multiclass_train_df):
    if n_instances > len(pool_indices):
        n_instances = len(pool_indices)  # Limit to available samples

    if n_instances == 0:
        return [], [], pool_indices

    selected_indices, remaining_indices = train_test_split(pool_indices, train_size=n_instances)
    X_add, y_add = X_pool[selected_indices], y_pool[selected_indices]
    sample_names = multiclass_train_df.iloc[selected_indices, 0].values.tolist()
    return X_add, y_add, remaining_indices, sample_names'''

def select_new_samples(pool_indices, X_pool_np, y_pool_np, n_instances, iteration, multiclass_train_df, samples_log_file):
    # Check if all remaining samples are requested in the last iteration
    if n_instances >= len(pool_indices):
        # Use all remaining pool indices
        selected_indices = pool_indices
        remaining_indices = []  # Pool will be empty after this selection
    else:
        # Use train_test_split for partial selection
        selected_indices, remaining_indices = train_test_split(pool_indices, train_size=n_instances)

    # Use the indices to extract the samples from X_pool_np and y_pool_np
    X_add = X_pool_np[selected_indices]
    y_add = y_pool_np[selected_indices]

    # Log the names of selected samples
    sample_names = multiclass_train_df.iloc[selected_indices, 0].values.tolist()
    with open(samples_log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([iteration + 1] + sample_names)

    return X_add, y_add, remaining_indices, sample_names

def save_checkpoint(classifier, iteration, path):
    checkpoint_path = f"{path}/model_checkpoint_iteration_{iteration}.pt"
    torch.save(classifier.module.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

def main_training_loop(classifier, X_train_initial_np, y_train_initial_np, X_pool_np, y_pool_np, X_test_np, y_test_np,
                       initial_df, multiclass_train_df, no_of_iterations, samples_log_file, checkpoint_path):
    pool_indices = list(range(len(X_pool_np)))
    POWER = 1.0
    f1_test_data = []

    for i in range(no_of_iterations):
        if i == 0:
            print(f"\nIteration {i + 1}: Using initial samples.")
            n_instances = len(X_train_initial_np)
            initial_sample_names = initial_df['image'].tolist()
            log_sample_names(i + 1, initial_sample_names, samples_log_file)
        else:
            if i == no_of_iterations - 1:
                n_instances = len(pool_indices)
                print(f"\nIteration {i + 1}: Requesting all remaining samples.")
            else:
                POWER += 0.25
                n_instances = batch(int(np.ceil(np.power(10, POWER))), batch_size)
                print(f"\nIteration {i + 1}: Requesting {n_instances} samples.")

            # Select and add new samples for the next training iteration
            if n_instances > 0:
                '''X_add, y_add, pool_indices, sample_names = select_new_samples(
                    pool_indices, X_pool_np, y_pool_np, n_instances, multiclass_train_df
                )'''
                X_add, y_add, pool_indices, sample_names = select_new_samples(
                    pool_indices, X_pool_np, y_pool_np, n_instances, i, multiclass_train_df, samples_log_file
                )
                if X_add.size > 0:
                    X_train_initial_np = np.concatenate((X_train_initial_np, X_add), axis=0)
                    y_train_initial_np = np.concatenate((y_train_initial_np, y_add), axis=0)
                    #log_sample_names(i + 1, sample_names, samples_log_file)

        # Train and evaluate model
        test_f1 = train_and_evaluate(classifier, X_train_initial_np, y_train_initial_np, X_test_np, y_test_np)
        f1_test_data.append(test_f1)
        print(f"Iteration {i + 1}: Test F1 Score: {test_f1}")

        # Save model checkpoint and clear GPU cache
        save_checkpoint(classifier, i, checkpoint_path)
        torch.cuda.empty_cache()

    return f1_test_data

# Run main training loop
f1_test_data = main_training_loop(
    classifier=classifier,
    X_train_initial_np=X_train_initial_np,
    y_train_initial_np=y_train_initial_np,
    X_pool_np=X_pool_np,
    y_pool_np=y_pool_np,
    X_test_np=X_test_np,
    y_test_np=y_test_np,
    initial_df=initial_df,
    multiclass_train_df=multiclass_train_df,
    no_of_iterations=no_of_iterations,
    samples_log_file=samples_log_file,
    checkpoint_path=save_dir
)





performance_test_data = [f1_test_data]

print('F1 Test Data:', f1_test_data)
#print('Accuracy Test Data:', acc_test_data)
print('Performance Test Data:', performance_test_data)


# Save performance metrics
'''performance_metrics_file = os.path.join(save_dir, "performance_metrics.csv")
with open(performance_metrics_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "F1 Score"])
    for iteration, (f1) in enumerate(zip(f1_test_data), start=1):
        writer.writerow([iteration, f1])'''
performance_metrics_file = os.path.join(save_dir, "performance_metrics.csv")
with open(performance_metrics_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "F1 Score"])
    for iteration, f1 in enumerate(f1_test_data, start=1):
        writer.writerow([iteration, f1])

