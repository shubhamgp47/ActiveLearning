"""
This module implements a Multi-Label Active Learning approach where no *new*
query strategy is used for sample selection. Instead, the images to be used at
each iteration have already been chosen by a Random Sampling Multi-Class
Active Learning process. 

Specifically, we:
1. Initialize our model using the images labeled as Iteration=1 in the CSV 
   (produced by the multiclass random sampling approach).
2. Conduct additional active learning “queries” by loading the images
   corresponding to Iterations 2, 3, …, and so forth from the same CSV.
3. At each iteration, we simply "teach" the Active Learner with these
   pre-selected images for multi-label classification—no new query strategy is 
   invoked here.

This script reads relevant hyperparameters and file paths from a config file
(`config.ini`), loads the DINO model for feature extraction, defines a Skorch
NeuralNetClassifier for multi-label classification, and logs performance
metrics such as Accuracy and F1 Score.
"""

import os
import gc
import csv
import random
import socket
import pickle
from io import StringIO
import configparser
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skorch.dataset import Dataset
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from skorch.helper import predefined_split
from skorch.classifier import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, Checkpoint, Callback, EpochScoring
from skorch.callbacks import LRScheduler


from sklearn.metrics import (
    f1_score,
    accuracy_score,
    make_scorer
)

from modAL.models import ActiveLearner

def find_config_file(filename='config.ini'):
    """
    Searches for the specified configuration file starting from the current
    directory and moving up the directory tree until the file is found.

    Args:
        filename (str): The name of the configuration file to find.

    Returns:
        str: The absolute path to the configuration file if found.

    Raises:
        FileNotFoundError: If the configuration file is not found in any parent directories.
    """
    current_dir = os.path.abspath(os.getcwd())
    while True:
        config_file_path = os.path.join(current_dir, filename)
        if os.path.isfile(config_file_path):
            return config_file_path
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"{filename} not found in any parent directories.")
        current_dir = parent_dir


config_path = find_config_file('config.ini')
config = configparser.ConfigParser()
config.read(config_path)

hostname = socket.gethostname()
if hostname == 'Power':
    paths = config['remote_2_paths']
    base_dir = os.path.abspath(config['save_path_remote2']['save_path'])
elif hostname == 'Kaiman':
    paths = config['remote_1_paths']
    base_dir = os.path.abspath(config['save_path_remote1']['save_path'])
elif hostname.startswith('tg') or hostname.startswith('tiny'):
    paths = config['hpc_paths']
    base_dir = os.path.abspath(config['save_path_hpc']['save_path'])
else:
    raise ValueError(f"Unknown hostname: {hostname}")


image_dir = os.path.abspath(paths['image_dir'])
train_df_path = os.path.abspath(paths['train_df'])
val_df_path = os.path.abspath(paths['val_df'])
test_df_path = os.path.abspath(paths['test_df'])


seed = int(config['seed_42']['seed'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


dropout = float(config['multilabel_DinoS_Parameters_test']['dropout'])
layer_freeze = config['multilabel_DinoS_Parameters_test']['layer_freeze']
criterion = eval(f"nn.{config['multilabel_DinoS_Parameters_test']['criterion']}")
batch_size = int(config['batch_size_8']['batch_size'])
optimizer_class = eval(f"optim.{config['multilabel_DinoS_Parameters_test']['optimizer']}")
lr = float(config['multilabel_DinoS_Parameters_test']['lr'])
patience = int(config['multilabel_DinoS_Parameters_test']['patience'])
step_size = int(config['multilabel_DinoS_Parameters_test']['step_size'])
gamma = float(config['multilabel_DinoS_Parameters_test']['gamma'])


N_QUERIES = 14
BEST_F1_SCORE = 0
SELECTED_IMAGES_CSV_PATH = 'RandomSampling/Multiclass/DinoS/seed_42'
FILENAME = "random_sampling_results_for_multilabel_classification_s42.pickle"
SAVE_PATH = 'RandomSampling/Multilabel_from_multiclass/DinoS/random_sampling_seed42_test'
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


selected_images_csv_dir = os.path.join(base_dir, SELECTED_IMAGES_CSV_PATH)

class CSVLogger(Callback):
    """
    Logs epoch data to a CSV file.

    Attributes:
        filename (str): Path to the CSV file.
        fieldnames (list): Column names for logging.
        file_exist (bool): Indicates whether the CSV file already exists.
    """

    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        self.file_exist = os.path.exists(filename)

    def on_epoch_end(self, net, **kwargs):
        """
        Called at the end of each epoch to log training/validation metrics.
        """
        logs = {key: net.history[-1, key] for key in self.fieldnames}
        if not self.file_exist:
            with open(self.filename, mode='w', newline='', encoding='utf-8') as file:
                csv_writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                csv_writer.writeheader()
            self.file_exist = True

        with open(self.filename, mode='a', newline='', encoding='utf-8') as file:
            csv_writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            csv_writer.writerow(logs)


class CustomDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding labels.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing image file names and labels.
        image_dir (str): Directory where images are stored.
        transform (callable, optional): Optional transform to be applied on an image sample.
    """

    def __init__(self, dataframe, image_directory, transformation=None):
        """
        Initializes the CustomDataset with a DataFrame, image directory, and optional transform.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image file names and labels.
            image_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on an image sample.
        """
        self.dataframe = dataframe
        self.image_directory = image_directory
        self.transformation = transformation

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a transformed image tensor
            and label is a tensor of labels.
        """
        img_name = os.path.join(self.image_directory, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transformation:
            image = self.transformation(image)
        label = self.dataframe.drop(columns=
                                    ['image', 'binary_NOK']).iloc[idx].values.astype('float32')
        label = torch.tensor(label).flatten()  # Ensure label is 1D
        return image, label


class CustomDINONormModel(nn.Module):
    """
    A custom neural network that integrates a pre-trained DINO model for
    feature extraction, along with additional layers tailored for multi-label classification.
    """

    def __init__(self, dino_model, num_classes=3):
        """
        Initializes the CustomDINONormModel with a pre-trained DINO model and a classifier.

        Args:
            dino_model (nn.Module): The pre-trained DINO model.
            num_classes (int): The number of output classes for the classifier.
            dropout (float): The dropout rate for the classifier.
            layer_freeze (str): The name of the layer up to which parameters should be frozen.
        """
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.freeze_layers(layer_freeze)


    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.dino_model(x)
        x = self.classifier(x)
        return x
    
    def freeze_layers(self, layer_name):
        """
        Freezes the layers of the DINO model up to the specified layer.

        Args:
            layer_name (str): The name of the layer up to which parameters should be frozen.
        """
        cutoff_reached = False
        for name, param in self.dino_model.named_parameters():
            if not cutoff_reached:
                param.requires_grad = False
                if layer_name in name:
                    cutoff_reached = True
            else:
                param.requires_grad = True


def tensors_to_numpy(loader):
    """
    Converts a batch of tensors from a DataLoader to numpy arrays.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing batches of tensors.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - images_array (np.ndarray): Numpy array of images with shape (N, H, W, C).
            - labels_array (np.ndarray): Numpy array of labels.
    """
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


def get_images_and_labels_for_iteration(
        selected_images_df, iteration, image_dir, label_df, transform=None):
    """
    Loads the images and their labels for a given iteration from a CSV that
    lists pre-selected images (Random Sampling).

    Parameters:
    selected_images_df (DataFrame): DataFrame containing the selected images.
    iteration (int): The current iteration number.
    image_dir (str): Directory where the images are stored.
    label_df (DataFrame): DataFrame containing the labels for the images.
    transform (callable, optional): Optional transform to be applied on an image.

    Returns:
    tuple: A tuple containing:
        - torch.Tensor: A tensor of loaded images.
        - torch.Tensor: A tensor of corresponding labels.

    Raises:
    ValueError: If no images are found for the given iteration or no valid images are found.
    """
    matching_rows = selected_images_df[selected_images_df["Query_Iteration"] == iteration]
    if matching_rows.empty:
        raise ValueError(f"No images found for Iteration {iteration}")

    image_list_str = matching_rows["Selected_Image"].values[0]
    image_list = image_list_str.split(',')

    loaded_images = []
    labels_list = []

    for img_name in image_list:
        img_name = img_name.strip()
        image_path = os.path.join(image_dir, img_name)
        if not os.path.exists(image_path):
            print(f"Warning: {img_name} not found in {image_dir}")
            continue

        img = Image.open(image_path).convert("RGB")
        if transform:
            img = transform(img)
        loaded_images.append(img)

        label_row = label_df[label_df["image"] == img_name][
            ['multi-label_double_winding', 'multi-label_gap', 'multi-label_crossing']
        ]
        if label_row.empty:
            print(f"Warning: No label found for {img_name} in label_df")
            continue
        labels_list.append(label_row.values[0])

    if not loaded_images:
        raise ValueError(f"No valid images found for Iteration {iteration}")

    return (
        torch.stack(loaded_images),
        torch.tensor(np.array(labels_list)).squeeze().float()
    )


train_df = pd.read_csv(train_df_path)
val_df = pd.read_csv(val_df_path)
test_df = pd.read_csv(test_df_path)

print("Train set shape:", train_df.shape)
print("Val set shape:  ", val_df.shape)
print("Test set shape: ", test_df.shape)

with open(os.path.join(selected_images_csv_dir, "sample_selection_log.csv"), 
          'r', encoding='utf-8') as file:
    lines = file.readlines()

processed_lines = []
for line in lines:
    parts = line.split(',', 1)
    if len(parts) == 2:
        processed_lines.append(parts[0] + ';' + parts[1].replace('\n', '').strip())
    else:
        processed_lines.append(line.strip())

processed_data = StringIO('\n'.join(processed_lines))
selected_images_df = pd.read_csv(processed_data, delimiter=';')

print("Selected images DataFrame (Random Sampling):")
print(selected_images_df.head())

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_dataset = CustomDataset(val_df, image_dir, transform)
test_dataset = CustomDataset(test_df, image_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

X_val_np, y_val_np = tensors_to_numpy(val_loader)
X_test_np, y_test_np = tensors_to_numpy(test_loader)

dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
model = CustomDINONormModel(dino_model, num_classes=3).to(DEVICE)

save_dir = os.path.join(base_dir, SAVE_PATH)
os.makedirs(save_dir, exist_ok=True)

fieldnames = ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']
csv_logger = CSVLogger(os.path.join(save_dir, "training_history.csv"), fieldnames)

f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)
early_stopper = EarlyStopping(monitor='valid_loss', patience=patience, lower_is_better=True)
cp = Checkpoint(dirname=os.path.join(save_dir, 'model_checkpoints'),
                monitor='valid_loss_best', f_params='best_model.pt')

lr_scheduler = LRScheduler(
    policy=lr_scheduler.StepLR,
    step_size=step_size,
    gamma=gamma
)

net = NeuralNetClassifier(
    module=model,
    criterion=nn.BCEWithLogitsLoss,
    optimizer=optimizer_class,
    lr=lr,
    max_epochs=100,
    train_split=predefined_split(
        Dataset(X_val_np, y_val_np)),
    device=DEVICE,
    callbacks=[train_f1, valid_f1, csv_logger, early_stopper, cp, lr_scheduler],
    verbose=1
)


X_query_initial, y_query_initial = get_images_and_labels_for_iteration(
    selected_images_df=selected_images_df,
    iteration=1,
    image_dir=image_dir,
    label_df=train_df,
    transform=transform
)


'''pre_logits = learner.predict(X_test_np)
pre_sigmoid = torch.sigmoid(torch.tensor(pre_logits)) > 0.5
pre_f1_score_macro = f1_score(y_test_np, pre_sigmoid.numpy(), average='macro')
pre_f1_score_micro = f1_score(y_test_np, pre_sigmoid.numpy(), average='micro')
pre_acc = accuracy_score(y_test_np, pre_sigmoid.numpy())

print("Pre-training scores with initial data:")
print(f"  F1 (macro) = {pre_f1_score_macro:.4f}")
print(f"  F1 (micro) = {pre_f1_score_micro:.4f}")
print(f"  Accuracy   = {pre_acc:.4f}")'''

f1_test_data = []
performance_val_data = []
performance_val_data_with_loss = []
f1_test_data_macro = []
performance_val_data_macro = []
no_of_samples = []
total_samples = X_query_initial.shape[0]
no_of_samples.append(total_samples)
acc_history = []

'''performance_val_data_with_loss.append(learner.estimator.history_)
f1_test_data.append(pre_f1_score_micro)
f1_test_data_macro.append(pre_f1_score_macro)
acc_history.append(pre_acc)'''

for i in range(N_QUERIES):
    print(f"\nQuery {i + 1}: Using the exact images and labels from the CSV for this iteration.")
    print(f"\nQuery {i + 1}: Using images from iteration={i + 1} (random sampling).")
    # Update the cumulative datasets
    if i == 0:
        X_cumulative = X_query_initial
        y_cumulative = y_query_initial
    else:
        # Load the exact images and their corresponding labels for this query iteration
        X_query, y_query = get_images_and_labels_for_iteration(selected_images_df=selected_images_df,
        iteration=i + 1, 
        image_dir=image_dir, 
        label_df=train_df, 
        transform=transform)

        # Convert to numpy
        X_query_np = X_query.numpy()
        y_query_np = y_query.numpy()

        X_cumulative = np.vstack((X_cumulative, X_query_np))
        y_cumulative = np.vstack((y_cumulative, y_query_np))  # Use vstack to match dimensions
    print(f"Number of samples used for training in Query {i + 1} is {len(X_cumulative)}")
    # Teach the learner with the new data
    net.fit(X=X_cumulative, y=y_cumulative)

    logits = net.predict(X_test_np)
    predictions = torch.sigmoid(torch.tensor(logits)) > 0.5
    acc = accuracy_score(y_test_np, predictions.numpy())
    f1_macro = f1_score(y_test_np, predictions.numpy(), average='macro')
    f1_micro = f1_score(y_test_np, predictions.numpy(), average='micro')

    y_pred_val = net.predict(X_val_np)
    val_accuracy = accuracy_score(y_val_np, y_pred_val)
    val_f1 = f1_score(y_val_np, y_pred_val, average='micro')
    val_f1_macro = f1_score(y_val_np, y_pred_val, average='macro')
    performance_val_data.append({'accuracy': val_accuracy, 'f1_micro': val_f1})
    performance_val_data_macro.append({'f1_macro': val_f1_macro})
    performance_val_data_with_loss.append(net.history_)
    f1_test_data.append(f1_micro)
    f1_test_data_macro.append(f1_macro)
    acc_history.append(acc)

    print(f"Iteration {i + 1} Test Accuracy: {acc:.4f}")
    print(f"Iteration {i + 1} Test F1 Score (micro): {f1_micro:.4f}")
    print(f"Iteration {i + 1} Test F1 Score (macro): {f1_macro:.4f}")

    if f1_macro > BEST_F1_SCORE:
        BEST_F1_SCORE = f1_macro

    checkpoint_path = os.path.join(save_dir, f"model_checkpoint_iteration_{i}.pt")
    torch.save(net.module_.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    best_model_path = os.path.join(
        save_dir, 'model_checkpoints', 'best_model.pt')

    net.load_params(best_model_path)

    gc.collect()
    torch.cuda.empty_cache()

print(f"\nBest F1 score across all iterations: {BEST_F1_SCORE:.4f}")

data_dict = {"test_f1_scores_micro": f1_test_data,
             "test_f1_scores_macro": f1_test_data_macro,
             "val_performance_with_loss": performance_val_data_with_loss,
             "val_performance_micro": performance_val_data,
             "val_performance_macro": performance_val_data_macro,
             "no_of_samples": no_of_samples,
             "accuracy" : acc_history
             }

file_path = os.path.join(save_dir, FILENAME)

with open(file_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Pickle file saved to {file_path}")
