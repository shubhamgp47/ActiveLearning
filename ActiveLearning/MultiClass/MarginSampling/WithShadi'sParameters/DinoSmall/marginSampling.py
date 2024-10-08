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
torch.multiprocessing.set_sharing_strategy('file_system')



# Define image directory and load dataframes
'''image_dir = os.path.abspath(r"D:/linear_winding_images_with_labels/")
IMAGES = os.path.abspath(r"C:/Users/localuserSK/Desktop/datasets/")
# Print all files in the directory to verify the presence of test.csv
print(os.listdir(IMAGES))
TRAIN_CSV_PATH = os.path.join(IMAGES, "train.csv")
train_df = pd.read_csv(TRAIN_CSV_PATH)
TEST_CSV_PATH = os.path.join(IMAGES, "test.csv")
test_df = pd.read_csv(TEST_CSV_PATH)
VAL_CSV_PATH = os.path.join(IMAGES, "validation.csv")
val_df = pd.read_csv(VAL_CSV_PATH)'''

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
    multiclass_labels.append(labels)                                # Collecting list of train, val and test labels in another list
    
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



# epochs = 100
batch_size =8
# layer_freeze= 69

# learning_rate = 0.1
# momentum_term = 0.2280969903050278

# dropout_rate = 0.49809628309801696
optimizer = 'SGD'


# Initialising Active learner and query strategy using modAL
from modAL.models import ActiveLearner
from modAL.uncertainty import margin_sampling

query_strategy = margin_sampling
# Function to find the nearest multiple of base for a given number
def batch(number, base):                                   
    return base * round(number / base)

# Directly set the training size based on the size of the dataset and batch size
# Calculate initial training size as 10% of the dataset
power=1
train_size = batch(int(np.ceil(np.power(10, power))), batch_size)  

print(f"Calculated train_size: {train_size}")


# Shuffling the data frame 
multiclass_train_df = multiclass_train_df.sample(frac = 1, random_state = 1234)

# Intial set must contain sample from each class
initial_df = multiclass_train_df.groupby('multiclass').head(1).head(train_size)

# Assigning remaining samples as the pool of data
pool_df = multiclass_train_df.drop(initial_df.index)




# Define the target size for resizing images
target_size = (224, 224)

# Initialize empty lists for storing images and labels
initial_train_image = []

# Define a transformation pipeline
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
# Load and preprocess images for initial training set
for i in tqdm(range(initial_df.shape[0])):
    image_path = os.path.join(image_dir, initial_df["image"].iloc[i])
    img = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    img = transform(img)
    initial_train_image.append(img)

# Convert the list of images to a PyTorch tensor and then permute
X_train_initial = torch.stack(initial_train_image).permute(0, 2, 3, 1)  # Move the channel to the last dimension

# Ensure the shape is [batch_size, 224, 224, 3]
print(f"Initial training data shape: {X_train_initial.shape}")

# Convert labels to one-hot encoding
y_multiclass = initial_df['multiclass'].astype(np.int64).to_numpy()  # Convert to NumPy array
num_classes = len(np.unique(y_multiclass))
y_train_initial = F.one_hot(torch.tensor(y_multiclass), num_classes=num_classes).float()

# Print the shape of the resulting tensors
print(f"X_train_initial shape: {X_train_initial.shape}, y_train_initial shape: {y_train_initial.shape}")
print(y_train_initial)




# Initialize empty lists for storing images and labels
pool_train_image = []

# Load and preprocess images in batches
for batch_start in range(0, pool_df.shape[0], batch_size):
    batch_end = min(batch_start + batch_size, pool_df.shape[0])
    batch_images = []

    for i in range(batch_start, batch_end):
        image_path = os.path.join(image_dir, pool_df["image"].iloc[i])
        img = Image.open(image_path).convert("RGB")  # Ensures the image is in RGB format
        img = transform(img)
        img_rgb = img.permute(1, 2, 0)  # Change shape from [C, H, W] to [H, W, C]
        batch_images.append(img_rgb)

    # Convert the list of images to a PyTorch tensor
    X_batch = torch.stack(batch_images)
    pool_train_image.append(X_batch)

# Concatenate all batches to create the final X_pool tensor
X_pool = torch.cat(pool_train_image)

# Convert multiclass labels to one-hot encoding and ensure proper shape
y_multiclass = pool_df['multiclass'].astype(np.int64).to_numpy()
num_classes = len(np.unique(y_multiclass))
y_pool_initial = F.one_hot(torch.tensor(y_multiclass), num_classes=num_classes).float()

# Ensure the tensor is in the shape [n_samples, n_classes]
y_pool_initial = y_pool_initial.squeeze()  # This removes any singleton dimensions if present


# Print the shape of the resulting tensors
print(f"X_pool shape: {X_pool.shape}, y_pool_initial shape: {y_pool_initial.shape}")




# Initialize empty lists for storing images and labels
val_images = []

# Load and preprocess images
for i in tqdm(range(multiclass_val_df.shape[0])):
    image_path = os.path.join(image_dir, multiclass_val_df["image"].iloc[i])
    img = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    img = transform(img)
    img_rgb = img.permute(1, 2, 0)  # Change shape from [C, H, W] to [H, W, C]
    val_images.append(img_rgb)

# Convert the list of images to a PyTorch tensor
X_val = torch.stack(val_images)

# Convert multiclass labels to a numpy array
y_multiclass = multiclass_val_df['multiclass'].astype(np.int64).to_numpy()
num_classes = len(np.unique(y_multiclass))
y_val = F.one_hot(torch.tensor(y_multiclass), num_classes=num_classes).float()

# Print the shape of the resulting tensors
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")



# Initialize empty lists for storing images and labels
test_images = []

# Load and preprocess images
for i in tqdm(range(multiclass_test_df.shape[0])):
    image_path = os.path.join(image_dir, multiclass_test_df["image"].iloc[i])
    img = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    img = transform(img)
    img_rgb = img.permute(1, 2, 0)  # Change shape from [C, H, W] to [H, W, C]
    test_images.append(img_rgb)

# Convert the list of images to a PyTorch tensor
X_test = torch.stack(test_images)

# Convert multiclass labels to a numpy array
y_multiclass = multiclass_test_df['multiclass'].astype(np.int64).to_numpy()
num_classes = len(np.unique(y_multiclass))

# Convert to one-hot encoding
y_test = F.one_hot(torch.tensor(y_multiclass), num_classes=num_classes).float()

# Print the shape of the resulting tensors
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
save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/DinoSmall/marginSampling/{run_id}"
os.makedirs(save_dir, exist_ok=True)

# Field names for the logger
fieldnames = ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']

# Initialize CSVLogger with the path and fieldnames
csv_logger = CSVLogger(os.path.join(save_dir, "training_history.csv"), fieldnames)

import os



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
from modAL.uncertainty import entropy_sampling


# Custom model class
# Custom model class
# Custom model class

# Custom model class
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, num_classes):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.09996859501133021),
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

# Load DINO model
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

for param in list(dino_model.parameters())[:65]:
    param.requires_grad = False

# Define hyperparameters
learning_rate = 0.002
momentum_term = 0.24729309193472406
dropout_rate = 0.49709490164030934
num_classes = 8
layer_freeze = 65

# Setup the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomDINONormModel(dino_model, num_classes=8).to(device)



from skorch.callbacks import EarlyStopping, Checkpoint, Callback

# Function to convert one-hot encoded labels to integer labels
def convert_one_hot_to_labels(y):
    return np.argmax(y, axis=1) if len(y.shape) > 1 else y

# Ensure that y_train_initial, y_pool_initial, y_val, and y_test are in the correct format
y_train_initial_np = convert_one_hot_to_labels(y_train_initial.clone().detach().cpu().numpy())
y_pool_initial_np = convert_one_hot_to_labels(y_pool_initial.clone().detach().cpu().numpy())
y_val_np = convert_one_hot_to_labels(y_val.clone().detach().cpu().numpy())
y_test_np = convert_one_hot_to_labels(y_test.clone().detach().cpu().numpy())

# Convert initial datasets to NumPy
X_train_initial_np = X_train_initial.clone().detach().cpu().numpy()
X_pool_np = X_pool.clone().detach().cpu().numpy()
X_val_np = X_val.clone().detach().cpu().numpy()
X_test_np = X_test.clone().detach().cpu().numpy()

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
es = EarlyStopping(monitor='valid_loss', patience=8, lower_is_better=True)
cp = Checkpoint(dirname='model_checkpoints', monitor='valid_loss_best')



classifier = NeuralNetClassifier(
    module=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.SGD,
    optimizer__momentum=momentum_term,
    lr=learning_rate,
    max_epochs=100,
    train_split=train_split,
    device=device,
    callbacks=[train_f1, valid_f1, es,csv_logger,cp],
    verbose=1
)



# Initialize the ActiveLearner
learner = ActiveLearner(
    estimator=classifier,
    query_strategy=margin_sampling,
    X_training=X_cumulative,
    y_training=y_cumulative
)

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
patience = 8  # Number of rounds to continue without improvement
wait = 0  # Current wait time

# Active learning loop parameters
n_queries = 13
initial_fraction = 0.1  # 10% of the dataset initially selected
start_point = 2000  # Initial number of instances selected
acc_test_data = []
f1_test_data = []
POWER=1

for i in range(n_queries):
    # Determine the number of samples to query
    if i == 12:
        n_instances = X_pool_np.shape[0]
    else:
        POWER += 0.25
        n_instances = batch(int(np.ceil(np.power(10, POWER))), batch_size)
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
    f1 = f1_score(y_test_np, y_pred, average='micro')
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
performance_filename = "/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/DinoSmall/marginSampling/performance_results_entropy_sampling.npy"
np.save(performance_filename, {"f1_scores": f1_test_data, "accuracies": acc_test_data})
print(f"Performance results saved to {performance_filename}")