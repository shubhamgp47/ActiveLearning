"""
This module implements Random Sampling using the.

The primary purpose of this module is to facilitate the active learning
process for multiclass classification tasks.
It includes functions and classes to load data,
train models, and evaluate performance using the average confidence
strategy to select the most informative samples for labeling.
"""
import os
import csv
import socket
import pickle
import configparser
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.hub
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
from PIL import Image
from skorch.callbacks import Callback, EarlyStopping, Checkpoint
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from skorch.callbacks import LRScheduler
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.model_selection import train_test_split



def find_config_file(filename='config.ini'):
    """
    Searches for the specified configuration file starting from the current directory
    and moving up the directory tree until the file is found.

    Args:
        filename (str): The name of the configuration file to find. Defaults to 'config.ini'.

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
            raise FileNotFoundError(
                f"{filename} not found in any parent directories")
        current_dir = parent_dir


config_path = find_config_file()
config = configparser.ConfigParser()
config.read(config_path)

hostname = socket.gethostname()
if hostname == 'Power':
    paths = config['remote_2_paths']
    base_dir = os.path.abspath(config['save_path_remote2']['save_path'])
elif hostname == 'Kaiman':
    paths = config['remote_1_paths']
    base_dir = os.path.abspath(config['save_path_remote1']['save_path'])
elif hostname.startswith("tg"):
    paths = config['hpc_paths']
    base_dir = os.path.abspath(config['save_path_hpc']['save_path'])
elif hostname.startswith("tin"):
    paths = config['hpc_paths']
    base_dir = os.path.abspath(config['save_path_hpc']['save_path'])
else:
    raise ValueError(f"Unknown hostname: {hostname}")

image_dir = os.path.abspath(paths['image_dir'])
train_df_path = os.path.abspath(paths['train_df'])
val_df_path = os.path.abspath(paths['val_df'])
test_df_path = os.path.abspath(paths['test_df'])

train_df = pd.read_csv(train_df_path)
val_df = pd.read_csv(val_df_path)
test_df = pd.read_csv(test_df_path)

seed = int(config['seed_43']['seed'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

predefined_image_names = config['predefined_images']['image_names'].split(', ')
batch_size = int(config['multiclass_EffNet_Parameters_3']['batch_size'])
dropout = float(config['multiclass_EffNet_Parameters_3']['dropout'])
layer_freeze = config['multiclass_EffNet_Parameters_3']['layer_freeze']
criterion = eval(f"nn.{config['multiclass_EffNet_Parameters_3']['criterion']}")
optimizer = eval(f"optim.{config['multiclass_EffNet_Parameters_3']['optimizer']}")
lr = float(config['multiclass_EffNet_Parameters_3']['lr'])
patience = int(config['multiclass_EffNet_Parameters_3']['patience'])
step_size = int(config['multiclass_EffNet_Parameters_3']['step_size'])
gamma = float(config['multiclass_EffNet_Parameters_3']['gamma'])

POWER = 1
NO_OF_ITERATIONS = 14
NUM_CLASSES = 8
FILENAME = "random_sampling_results_for_multiclass_classification_s43.pickle"
SPECIFIC_PATH = 'RandomSampling/Multiclass/EfficientNet/seed_43'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

list_data_frame = [train_df, test_df, val_df]
multiclass_labels = []

for x in range(len(list_data_frame)):
    labels = []
    for y in tqdm(range(list_data_frame[x].shape[0])):
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

multiclass_train_df = train_df.assign(multiclass = multiclass_labels[0])
multiclass_train_df = multiclass_train_df[['image', 'multiclass']].dropna()

multiclass_test_df = test_df.assign(multiclass = multiclass_labels[1])
multiclass_test_df = multiclass_test_df[['image', 'multiclass']].dropna()

multiclass_val_df = val_df.assign(multiclass = multiclass_labels[2])
multiclass_val_df = multiclass_val_df[['image', 'multiclass']].dropna()

print("multiclass_train_df shape:", multiclass_train_df.shape)
print("multiclass_test_df shape:", multiclass_test_df.shape)
print("multiclass_val_df shape:", multiclass_val_df.shape)

def batch(number, base):
    """
    Rounds a given number to the nearest multiple of a specified base.

    This function takes a number and a base, and returns the nearest multiple of the base
    by rounding the result of the division of the number by the base.

    Parameters:
    number (float or int): The number to be rounded.
    base (float or int): The base to which the number will be rounded.

    Returns:
    float or int: The nearest multiple of the base.

    Example:
    >>> batch(23, 5)
    25
    >>> batch(17, 3)
    18
    """
    multiple = base * round(np.divide(number, base))
    return multiple

train_size = batch(int(np.ceil(np.power(10, POWER))), batch_size)
print(f"Calculated train_size: {train_size}")

multiclass_train_df = multiclass_train_df.sample(frac=1, random_state=1234)

initial_df = multiclass_train_df[multiclass_train_df['image'].isin(
    predefined_image_names)]
initial_df = initial_df.set_index(
    'image').loc[predefined_image_names].reset_index()

missing_images = [
    img for img in predefined_image_names if img not in multiclass_train_df['image'].values]
if missing_images:
    print("These images are missing from multiclass_train_df:", missing_images)
    assert not missing_images, "Initial DataFrame doesn't contain the specified image."

pool_df = multiclass_train_df.drop(initial_df.index)


class CustomDataset(Dataset):
    """
    A custom dataset class for loading images and their corresponding labels from a dataframe.

    Attributes:
        dataframe (pd.DataFrame): A dataframe containing image file names and labels.
        image_directory (str): The directory where the images are stored.
        transformation (callable, optional): A function/transform to apply to the images.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the image and label at the specified index.

    Args:
        dataframe (pd.DataFrame): A dataframe containing image file names and labels.
        image_directory (str): The directory where the images are stored.
        transformation (callable, optional): A function/transform to apply to the images.
    """

    def __init__(self, dataframe, image_directory, transformation=None):
        self.dataframe = dataframe
        self.image_directory = image_directory
        self.transformation = transformation

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_directory,
                                self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transformation:
            image = self.transformation(image)
        label = int(self.dataframe.iloc[idx, 1])
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


initial_dataset = CustomDataset(initial_df, image_dir, transform)
train_dataset = CustomDataset(multiclass_train_df, image_dir, transform)
pool_dataset = CustomDataset(pool_df, image_dir, transform)
val_dataset = CustomDataset(multiclass_val_df, image_dir, transform)
test_dataset = CustomDataset(multiclass_test_df, image_dir, transform)

initial_loader = DataLoader(
    initial_dataset, batch_size=batch_size, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


class CSVLogger(Callback):
    """
    Log epoch data to a CSV file.

    Attributes:
        filename (str): The name of the CSV file to log data to.
        fieldnames (list): The list of field names for the CSV file.
        file_exist (bool): Flag indicating whether the CSV file already exists.
    """

    def __init__(self, filename, fieldnames):
        """
        Initializes the CSVLogger with a filename and fieldnames.

        Args:
            filename (str): The name of the CSV file to log data to.
            fieldnames (list): The list of field names for the CSV file.
        """
        super().__init__()
        self.filename = filename
        self.fieldnames = fieldnames
        self.file_exist = os.path.exists(filename)

    def on_epoch_end(self, net, **kwargs):
        """
        Called at the end of each epoch to log data to the CSV file.

        Args:
            net: The neural network being trained.
            **kwargs: Additional keyword arguments.
        """
        logs = {key: net.history[-1, key] for key in self.fieldnames}
        if not self.file_exist:
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            self.file_exist = True
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(logs)


save_dir = os.path.join(base_dir, SPECIFIC_PATH)
os.makedirs(save_dir, exist_ok=True)

csv_logger = CSVLogger(
    os.path.join(save_dir, "training_history.csv"),
    ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']
)

class CustomEfficientNetNormModel(nn.Module):
    """
    Custom neural network model that integrates an EfficientNet V2 Large backbone with additional classification layers.
    """

    def __init__(self, efficientnet_model: nn.Module, dropout_rate: float, num_classes: int, num_ftrs: int):
        """
        Initializes the CustomEfficientNetNormModel with an EfficientNet backbone and a classifier head.

        Args:
            efficientnet_model (nn.Module): Pretrained EfficientNet model.
            dropout_rate (float): Dropout rate for the classifier.
            num_classes (int): Number of output classes for classification.
            num_ftrs (int): Number of input features to the classifier.
        """
        super(CustomEfficientNetNormModel, self).__init__()
        self.efficientnet_model = efficientnet_model
        # Replace the classifier with a custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model, including input shape handling.

        Args:
            x (torch.Tensor): Input tensor of shape (N, H, W, C) or (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor.
        """
        # Handle input shape (N, H, W, C)
        if x.dim() == 4 and x.shape[3] == 3:  # (N, H, W, C)
            x = x.permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
        elif x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Unexpected input dimensions: {x.shape}")

        # Pass through EfficientNet
        x = self.efficientnet_model.features(x)
        x = self.efficientnet_model.avgpool(x)
        x = torch.flatten(x, 1)

        # Pass through classifier
        x = self.classifier(x)
        return x


LAYER_FREEZE_OPTIONS = [
    'efficientnet.features.0',  # ConvNormActivation
    'efficientnet.features.1',  # MBConv Block 1
    'efficientnet.features.2',  # MBConv Block 2
    'efficientnet.features.3',  # MBConv Block 3
    'efficientnet.features.4',  # MBConv Block 4
    'efficientnet.features.5',  # MBConv Block 5
    'efficientnet.features.6',  # MBConv Block 6
    'efficientnet.features.7',  # MBConv Block 7
    'efficientnet.features.8'   # MBConv Block 8
]

def freeze_layers_by_name(efficientnet_model: nn.Module, freeze_upto_name: str) -> None:
    """
    Freezes the layers of the EfficientNet model up to and including the specified layer name.

    Args:
        efficientnet_model (nn.Module): The EfficientNet model whose layers are to be frozen.
        freeze_upto_name (str): The block name to freeze up to, e.g., 'efficientnet.features.5'.
    """
    # Unfreeze all parameters initially
    for param in efficientnet_model.parameters():
        param.requires_grad = True

    # Find the block index corresponding to the freeze_upto_name
    try:
        freeze_index = LAYER_FREEZE_OPTIONS.index(freeze_upto_name)
        #freeze_index = layer_freeze
    except ValueError:
        raise ValueError(f"Layer name '{freeze_upto_name}' not found in LAYER_FREEZE_OPTIONS.")

    # Freeze layers from block 0 up to freeze_index (inclusive)
    for block_idx in range(freeze_index + 1):
        for param in efficientnet_model.features[block_idx].parameters():
            param.requires_grad = False

# 7) DEFINE FUNCTION TO INITIALIZE THE MODEL

def define_model(dropout: float, freeze_upto_name: str, num_classes: int) -> nn.Module:
    """
    Defines the model by loading the EfficientNet backbone, freezing specified layers, and attaching the classifier head.

    Args:
        dropout (float): Dropout rate for the classifier.
        freeze_upto_name (str): The layer name up to which layers are frozen.
        num_classes (int): Number of output classes for classification.

    Returns:
        nn.Module: The complete model ready for training.
    """
    torch.cuda.empty_cache()

    # Load the EfficientNet V2 Large model with pretrained weights
    efficientnet_model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

    # Freeze layers up to the specified boundary
    freeze_layers_by_name(efficientnet_model, freeze_upto_name)

    # Retrieve the number of input features to the classifier
    num_ftrs = efficientnet_model.classifier[1].in_features  # Typically 1408 for EfficientNet V2 Large

    # Remove the existing classifier to get feature representations
    efficientnet_model.classifier = nn.Identity()

    # Create the custom classification model
    model = CustomEfficientNetNormModel(
        efficientnet_model=efficientnet_model,
        dropout_rate=dropout,
        num_classes=NUM_CLASSES,
        num_ftrs=num_ftrs
    ).to(DEVICE)

    return model

effnet_model = define_model(dropout, layer_freeze, NUM_CLASSES)

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
    images_array, labels_array = [], []
    for data in loader:
        batch_images = data[0].numpy()
        batch_labels = data[1].numpy()
        batch_images = batch_images.transpose(0, 2, 3, 1)
        images_array.append(batch_images)
        labels_array.append(batch_labels)
    images_array = np.concatenate(images_array, axis=0)
    labels_array = np.concatenate(labels_array, axis=0)
    return images_array, labels_array


X_initial_np, y_initial_np = tensors_to_numpy(initial_loader)
X_train_np, y_train_np = tensors_to_numpy(train_loader)
X_pool_np, y_pool_np = tensors_to_numpy(pool_loader)
X_test_np, y_test_np = tensors_to_numpy(test_loader)
X_val_np, y_val_np = tensors_to_numpy(val_loader)

f1_scorer = make_scorer(f1_score, average='micro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True,
                        name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False,
                        name='valid_f1', lower_is_better=False)

valid_ds = Dataset(X_val_np, y_val_np)
train_split = predefined_split(valid_ds)
es = EarlyStopping(monitor='valid_loss', patience=patience, lower_is_better=True)
cp = Checkpoint(dirname=os.path.join(save_dir, 'model_checkpoints'),
                monitor='valid_loss_best', f_params='best_model.pt')

lr_scheduler = LRScheduler(
    policy=lr_scheduler.StepLR,
    step_size=step_size,
    gamma=gamma
)

classifier = NeuralNetClassifier(
    module=effnet_model,
    criterion=criterion,
    optimizer=optimizer,
    batch_size=batch_size,
    lr=lr,
    max_epochs=100,
    train_split=predefined_split(
        Dataset(X_val_np, y_val_np)),
    device=DEVICE,
    callbacks=[train_f1, valid_f1, es, cp, csv_logger,
               lr_scheduler],
    verbose=1
)

f1_test_data = []
performance_val_data = []
performance_val_data_with_loss = []
f1_test_data_macro = []
performance_val_data_macro = []
no_of_samples = []
total_samples = X_initial_np.shape[0]
no_of_samples.append(total_samples)


samples_log_file = os.path.join(save_dir, "sample_selection_log.csv")
with open(samples_log_file, mode='w', newline='', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Query_Iteration", "Selected_Image"])

X_train_initial_np = X_initial_np
y_train_initial_np = y_initial_np


def train_and_evaluate(classifier_model, x_train, y_train, x_test, y_test):
    """
    Trains the classifier on the training data and evaluates it on the test data.

    This function fits the provided classifier to the training data (X_train, y_train),
    makes predictions on the test data (X_test), and calculates the F1 scores.

    Parameters:
    classifier (object): The classifier to be trained and evaluated.
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    X_test (array-like): Test data features.
    y_test (array-like): Test data labels.

    Returns:
    tuple: A tuple containing the micro-averaged F1 score and the macro-averaged F1 score.
        - test_f1 (float): Micro-averaged F1 score on the test data.
        - test_f1_macro (float): Macro-averaged F1 score on the test data.

    Example:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import f1_score
    >>> classifier = RandomForestClassifier()
    >>> X_train, y_train = [[0, 1], [1, 0]], [0, 1]
    >>> X_test, y_test = [[0, 1], [1, 0]], [0, 1]
    >>> train_and_evaluate(classifier, X_train, y_train, X_test, y_test)
    (1.0, 1.0)
    """
    classifier_model.fit(x_train, y_train)
    y_pred = classifier_model.predict(x_test)
    test_f1 = f1_score(y_test, y_pred, average='micro')
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    return test_f1, test_f1_macro


def log_sample_names(iteration, sample_names, log_file):
    """
    Logs the sample names along with the iteration number to a specified log file.

    This function appends the iteration number and the list of sample names to a CSV file.
    Each row in the CSV file will contain the iteration number followed by the sample names.

    Parameters:
    iteration (int): The current iteration number.
    sample_names (list): A list of sample names to be logged.
    log_file (str): The path to the log file where the data will be appended.

    Example:
    >>> log_sample_names(1, ['sample1', 'sample2', 'sample3'], 'log.csv')
    This will append the following row to 'log.csv':
    1, sample1, sample2, sample3
    """
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([iteration] + sample_names)


def select_new_samples(pool_indexes, x_pool_np, y_pool_np_param, n_instances, iteration, train_df_param, samples_log_file_param):
    """
    Selects new samples for training and logs the sample names.

    This function selects a specified number of new samples from the pool of unlabeled data,
    adds them to the training set, and logs the sample names along with the iteration number
    to a specified log file. If the number of instances to select is greater than or equal to
    the number of available pool indices, all pool indices are selected.

    Parameters:
    pool_indexes (array-like): Indices of the pool of unlabeled data.
    x_pool_np (array-like): Features of the pool of unlabeled data.
    y_pool_np_param (array-like): Labels of the pool of unlabeled data.
    n_instances (int): Number of instances to select.
    iteration (int): The current iteration number.
    train_df (DataFrame): DataFrame containing the training data with sample names.
    samples_log_file_param (str): Path to the log file where sample names will be appended.

    Returns:
    tuple: A tuple containing the following elements:
        - X_add (array-like): Features of the selected samples.
        - y_add (array-like): Labels of the selected samples.
        - remaining_indices (array-like): Indices of the remaining pool of unlabeled data.
        - sample_names (list): List of selected sample names.

    Example:
    >>> pool_indexes = [0, 1, 2, 3, 4]
    >>> x_pool_np = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    >>> y_pool_np_param = np.array([0, 1, 0, 1, 0])
    >>> n_instances = 2
    >>> iteration = 1
    >>> train_df_param = pd.DataFrame({'image': ['img1', 'img2', 'img3', 'img4', 'img5']})
    >>> samples_log_file_param = 'samples_log.csv'
    >>> select_new_samples(pool_indexes, x_pool_np, y_pool_np_param, n_instances, iteration, train_df_param, samples_log_file_param)
    (array([[1, 2], [3, 4]]), array([0, 1]), array([2, 3, 4]), ['img1', 'img2'])
    """
    if n_instances >= len(pool_indexes):
        selected_indices = pool_indexes
        remaining_indices = []
    else:
        selected_indices, remaining_indices = train_test_split(
            pool_indexes, train_size=n_instances)

    x_add = x_pool_np[selected_indices]
    y_add = y_pool_np_param[selected_indices]

    sample_names = train_df_param.iloc[selected_indices, 0].values.tolist()
    with open(samples_log_file_param, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([iteration + 1] + sample_names)

    return x_add, y_add, remaining_indices, sample_names


def save_checkpoint(classifier_model, iteration, path):
    """
    Saves a checkpoint of the classifier's state at a specified iteration.

    This function saves the state dictionary of the classifier's module to a file,
    creating a checkpoint that can be used to resume training or for evaluation purposes.
    The checkpoint file is named based on the provided iteration number.

    Parameters:
    classifier_model (object): The classifier whose state is to be saved.
    iteration (int): The current iteration number.
    path (str): The directory path where the checkpoint file will be saved.

    Example:
    >>> save_checkpoint(classifier_model, 10, '/path/to/checkpoints')
    This will save the checkpoint file as '/path/to/checkpoints/model_checkpoint_iteration_10.pt'
    """
    checkpoint_path = f"{path}/model_checkpoint_iteration_{iteration}.pt"
    torch.save(classifier_model.module.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")


def main_training_loop(classifier_model, x_train_initial_np, y_train_initial_np_param, x_pool_np, y_pool_np_param, x_test_np, y_test_np_param,
                       initial_df_param, train_df_param, no_of_iterations, samples_log_file_param, checkpoint_path):
    """
    Executes the main training loop for active learning with a classifier.

    This function performs multiple iterations of training and evaluation, selecting new samples
    from the pool of unlabeled data, logging sample names, and saving model checkpoints. It also
    tracks performance metrics such as F1 scores and validation accuracy.

    Parameters:
    classifier_model (object): The classifier to be trained and evaluated.
    x_train_initial_np (array-like): Initial training data features.
    y_train_initial_np_param (array-like): Initial training data labels.
    x_pool_np (array-like): Features of the pool of unlabeled data.
    y_pool_np_param (array-like): Labels of the pool of unlabeled data.
    x_test_np (array-like): Test data features.
    y_test_np_param (array-like): Test data labels.
    initial_df_param (DataFrame): DataFrame containing initial training data with sample names.
    train_df_param (DataFrame): DataFrame containing training data with sample names.
    no_of_iterations (int): Number of iterations for the training loop.
    samples_log_file_param (str): Path to the log file where sample names will be appended.
    checkpoint_path (str): Directory path where model checkpoints will be saved.

    Returns:
    tuple: A tuple containing the following elements:
        - f1_test_data (list): List of micro-averaged F1 scores for each iteration.
        - f1_test_data_macro (list): List of macro-averaged F1 scores for each iteration.
        - performance_val_data (list): List of dictionaries with validation accuracy and micro-averaged F1 scores.
        - performance_val_data_macro (list): List of dictionaries with macro-averaged F1 scores.
        - performance_val_data_with_loss (list): List of classifier history objects for each iteration.
        - no_of_samples (list): List of the total number of samples used in each iteration.

    Example:
    >>> classifier_model = YourClassifier()
    >>> X_train_initial_np = np.array([[1, 2], [3, 4]])
    >>> y_train_initial_np_param = np.array([0, 1])
    >>> x_pool_np = np.array([[5, 6], [7, 8], [9, 10]])
    >>> y_pool_np_param = np.array([0, 1, 0])
    >>> x_test_np = np.array([[11, 12], [13, 14]])
    >>> y_test_np_param = np.array([0, 1])
    >>> initial_df_param = pd.DataFrame({'image': ['img1', 'img2']})
    >>> train_df_param = pd.DataFrame({'image': ['img3', 'img4', 'img5']})
    >>> no_of_iterations = 5
    >>> samples_log_file_param = 'samples_log.csv'
    >>> checkpoint_path = '/path/to/checkpoints'
    >>> main_training_loop(classifier_model, X_train_initial_np, y_train_initial_np_param, x_pool_np, y_pool_np_param, x_test_np, y_test_np_param,
                           initial_df_param, train_df_param, no_of_iterations, samples_log_file_param, checkpoint_path)
    """
    global POWER
    pool_indices = list(range(len(x_pool_np)))
    total_samples = len(x_train_initial_np)
    #no_of_samples.append(total_samples)

    for i in range(NO_OF_ITERATIONS):
        if i == 0:
            print(f"\nIteration {i + 1}: Using initial samples.")
            n_instances = len(x_train_initial_np)
            initial_sample_names = initial_df_param['image'].tolist()
            log_sample_names(i + 1, initial_sample_names,
                             samples_log_file_param)
        else:
            if i == no_of_iterations - 1:
                n_instances = len(pool_indices)
                print(
                    f"\nIteration {i + 1}: Requesting all remaining samples.")
            else:
                POWER += 0.25
                n_instances = batch(
                    int(np.ceil(np.power(10, POWER))), batch_size)
                print(
                    f"\nIteration {i + 1}: Requesting {n_instances} samples.")

            if n_instances > 0:
                x_add, y_add, pool_indices, _ = select_new_samples(
                    pool_indices, x_pool_np, y_pool_np_param, n_instances, i, train_df_param, samples_log_file_param
                )
                if x_add.size > 0:
                    x_train_initial_np = np.concatenate(
                        (x_train_initial_np, x_add), axis=0)
                    y_train_initial_np_param = np.concatenate(
                        (y_train_initial_np_param, y_add), axis=0)

        test_f1, test_f1_macro = train_and_evaluate(
            classifier_model, x_train_initial_np, y_train_initial_np_param, x_test_np, y_test_np_param)
        f1_test_data.append(test_f1)
        f1_test_data_macro.append(test_f1_macro)
        print(f"Iteration {i + 1}: Test F1 Micro Score: {test_f1}")
        print(f"Iteration {i + 1}: Test F1 Macro Score: {test_f1_macro}")

        save_checkpoint(classifier_model, i, checkpoint_path)
        torch.cuda.empty_cache()

        y_pred_val = classifier_model.predict(X_val_np)
        val_accuracy = accuracy_score(y_val_np, y_pred_val)
        val_f1 = f1_score(y_val_np, y_pred_val, average='micro')
        val_f1_macro = f1_score(y_val_np, y_pred_val, average='macro')
        performance_val_data.append(
            {'accuracy': val_accuracy, 'f1_micro': val_f1})
        performance_val_data_macro.append({'f1_macro': val_f1_macro})
        performance_val_data_with_loss.append(classifier_model.history_)

        total_samples += n_instances
        no_of_samples.append(total_samples)

        best_model_path = os.path.join(
            save_dir, 'model_checkpoints', 'best_model.pt')

        classifier.load_params(best_model_path)
    return f1_test_data, f1_test_data_macro, performance_val_data, performance_val_data_macro, performance_val_data_with_loss, no_of_samples


f1_test_data, f1_test_data_macro, performance_val_data, performance_val_data_macro, \
    performance_val_data_with_loss, no_of_samples = main_training_loop(
        classifier_model=classifier,
        x_train_initial_np=X_train_initial_np,
        y_train_initial_np_param=y_train_initial_np,
        x_pool_np=X_pool_np,
        y_pool_np_param=y_pool_np,
        x_test_np=X_test_np,
        y_test_np_param=y_test_np,
        initial_df_param=initial_df,
        train_df_param=train_df,
        no_of_iterations=NO_OF_ITERATIONS,
        samples_log_file_param=samples_log_file,
        checkpoint_path=save_dir
    )


data_dict = {
    "test_f1_scores_micro": f1_test_data,
    "test_f1_scores_macro": f1_test_data_macro,
    "val_performance_with_loss": performance_val_data_with_loss,
    "val_performance_micro": performance_val_data,
    "val_performance_macro": performance_val_data_macro,
    "no_of_samples": no_of_samples
}

file_path = os.path.join(save_dir, FILENAME)

with open(file_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Pickle file saved to {file_path}")