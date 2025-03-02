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
import random

seed_value = 42
#random_state = np.random.RandomState(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Directory and CSV setup
'''IMAGE_DIR = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")
TRAIN_CSV_PATH = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/Splits_v2024-03-18/train_v2024-03-18_10%.csv")
train_df = pd.read_csv(TRAIN_CSV_PATH)
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
TEST_CSV_PATH = os.path.join(df_dir, "test_v2024-03-18.csv")
test_df = pd.read_csv(TEST_CSV_PATH)
VAL_CSV_PATH = os.path.join(df_dir, "validation_v2024-03-18.csv")
val_df = pd.read_csv(VAL_CSV_PATH)'''

#torch.multiprocessing.set_sharing_strategy('file_system')

'''# Directory and CSV setup
image_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")
#df_dir_25 = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/Splits_v2024-03-18/")
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")
SELECTED_IMAGES_CSV_PATH = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/multilabel/RandomSampling/imagesfrommulticlass/DinoSmall/")
selected_images_df = pd.read_csv(SELECTED_IMAGES_CSV_PATH + "/sample_selection_log.csv")'''

image_dir = os.path.abspath('D:/linear_winding_images_with_labels/')
df_dir = os.path.abspath('D:/datasets/')
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")
SELECTED_IMAGES_CSV_PATH = os.path.abspath("C:/Users/localuserSG/ActiveLearning/Multilabel01/DinoSmall/imagesFromMulticlass/RandomSampling/")  
selected_images_df = pd.read_csv(SELECTED_IMAGES_CSV_PATH + "/sample_selection_log_seed_resultof_MLParams.csv")



def batch(number, base):
    return base * round(number / base)



# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
        label = self.dataframe.drop(columns=['image', 'binary_NOK']).iloc[idx].values.astype('float32')
        label = torch.tensor(label)  # Ensure label is a tensor
        return image, label

batch_size = 8
val_dataset = CustomDataset(val_df, image_dir, transform)
test_dataset = CustomDataset(test_df, image_dir, transform)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Adjust batch size
power = 1
train_size = int(np.ceil(np.power(10, power)))

# Split datasets
#initial_dataset, pool_dataset = split_datasets(X_train_initial, y_train_initial, train_size)

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
#save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/MultiLabel/RandomSampling_dataloadfix/{run_id}"
#save_dir = os.path.abspath('D:/Shubham/results/multilabel01/DinoSmall/RandomSampling/withseed/imagesFroMullticlass/')
save_dir = os.path.abspath('D:/Shubham/results/multilabel01/DinoSmall/experiment/RS/')
os.makedirs(save_dir, exist_ok=True)

# Field names for the logger
fieldnames = ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']
csv_logger = CSVLogger(os.path.join(save_dir, "training_history.csv"), fieldnames)

# Set proxy if necessary
os.environ['http_proxy'] = 'http://proxy:80'
os.environ['https_proxy'] = 'http://proxy:80'

# Custom model class
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, num_classes=3):  # Ensure num_classes=3
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.49709490164030934),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, num_classes)  # Output 3 classes
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

model = CustomDINONormModel(dino_model, num_classes=3).to(device)

# Function to convert tensors to numpy
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


X_test_np, y_test_np = tensors_to_numpy(test_loader)
X_val_np, y_val_np = tensors_to_numpy(val_loader)

# Skorch callbacks and classifier setup
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)
es = EarlyStopping(monitor='valid_loss', patience=13, lower_is_better=True)
model_checkpoint = Checkpoint(dirname=os.path.join(save_dir, 'model_checkpoints'), monitor='valid_loss_best')

def get_images_and_labels_for_iteration(selected_images_df, iteration, image_dir, label_df):
    # Get the selected images for this iteration as a single string
    images_for_iteration = selected_images_df[selected_images_df["Query_Iteration"] == iteration]["Selected_Image"].values[0]
    
    # Split the images string by semicolon to get individual filenames
    image_list = images_for_iteration.split(';')

    loaded_images = []
    labels_list = []  # Use a list for accumulating labels

    for img_name in image_list:
        img_name = img_name.strip()  # Remove any leading/trailing whitespace
        image_path = os.path.join(image_dir, img_name)
        
        # Load and transform the image
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        loaded_images.append(img)

        # Fetch the corresponding labels from the label_df based on the image name
        label = label_df[label_df["image"] == img_name][['multi-label_double_winding', 'multi-label_gap', 'multi-label_crossing']].values[0]
        labels_list.append(label)  # Append the label to the list

    # Convert lists to NumPy arrays and then to torch tensors
    loaded_images = torch.stack(loaded_images)
    labels_array = np.array(labels_list)  # Convert labels list to NumPy array

    return loaded_images, torch.tensor(labels_array).squeeze().float()


X_query_initial, y_query_initial = get_images_and_labels_for_iteration(selected_images_df, iteration=1, image_dir=image_dir, label_df=train_df)

estimator=NeuralNetClassifier(
        module=model,
        criterion=nn.BCEWithLogitsLoss,  # Use BCEWithLogitsLoss for multi-label classification
        optimizer=optim.SGD,
        optimizer__momentum=0.14729309193472406,
        lr=0.0001375803586556554,
        max_epochs=100,
        train_split=predefined_split(Dataset(X_val_np, y_val_np)),  # Validation set split
        device=device,
        callbacks=[train_f1, valid_f1, csv_logger, es]
    )

n_queries = 14
best_f1_score = 0
wait = 0
patience = 13
acc_test_data = []
f1_test_data = []


# Active Learning Loop with Corrected Label Handling
for i in range(n_queries):
    print(f"\nQuery {i + 1}: Using the exact images and labels from the CSV for this iteration.")
    # Update the cumulative datasets
    if i == 0:
        X_cumulative = X_query_initial
        y_cumulative = y_query_initial
    else:
        # Load the exact images and their corresponding labels for this query iteration
        X_query, y_query = get_images_and_labels_for_iteration(selected_images_df, iteration=i + 1, image_dir=image_dir, label_df=train_df)

        # Convert to numpy
        X_query_np = X_query.numpy()
        y_query_np = y_query.numpy()

        X_cumulative = np.vstack((X_cumulative, X_query_np))
        y_cumulative = np.vstack((y_cumulative, y_query_np))  # Use vstack to match dimensions
    print(f"Number of samples used for training in Query {i + 1} is {len(X_cumulative)}")
    # Teach the learner with the new data
    estimator.fit(X=X_cumulative, y=y_cumulative)

    # Evaluate on test set
    y_pred = estimator.predict(X_test_np)
    y_pred = torch.sigmoid(torch.tensor(y_pred)) > 0.5  # Use thresholding for multi-label prediction
    accuracy = accuracy_score(y_test_np, y_pred.numpy())
    f1 = f1_score(y_test_np, y_pred.numpy(), average='macro')
    acc_test_data.append(accuracy)
    f1_test_data.append(f1)

    print(f"Accuracy after query {i + 1}: {accuracy}")
    print(f"F1 Score after query {i + 1}: {f1}")

    if f1 > best_f1_score:
        best_f1_score = f1

    checkpoint_path = os.path.join(save_dir, f"model_checkpoint_iteration_{i}.pt")
    torch.save(estimator.module_.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

print(f"Best F1 score across iterations: {best_f1_score}")

# Save performance metrics
performance_metrics_file = os.path.join(save_dir, "performance_metrics.csv")
with open(performance_metrics_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Accuracy", "F1 Score"])
    for iteration, (acc, f1) in enumerate(zip(acc_test_data, f1_test_data), start=1):
        writer.writerow([iteration, acc, f1])