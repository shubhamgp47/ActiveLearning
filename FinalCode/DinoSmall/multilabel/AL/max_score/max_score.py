"""
This module implements Multilabel Active Learning using the Average Score strategy.

The primary purpose of this module is to facilitate the active learning
process for multilabel classification tasks.
It includes functions and classes to load data,
train models, and evaluate performance using the average confidence
strategy to select the most informative samples for labeling.
"""
import os
import csv
import socket
import configparser
import random
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from skorch.callbacks import Callback, EarlyStopping, Checkpoint
from skorch.classifier import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from modAL.models import ActiveLearner
from modAL.multilabel import max_score
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from torch.utils.data import DataLoader
from skorch.callbacks import LRScheduler
import torch.optim.lr_scheduler as lr_scheduler

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
            raise FileNotFoundError(f"{filename} not found in any parent directories")
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
elif hostname.startswith('tg'):
    paths = config['hpc_paths']
    base_dir = os.path.abspath(config['save_path_hpc']['save_path'])
elif hostname.startswith('tiny'):
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

seed = int(config['seed_45']['seed'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

predefined_image_names = config['predefined_images']['image_names'].split(', ')
dropout = float(config['multilabel_DinoS_Parameters_2']['dropout'])
layer_freeze = config['multilabel_DinoS_Parameters_2']['layer_freeze']
criterion = eval(f"nn.{config['multilabel_DinoS_Parameters_2']['criterion']}")
batch_size = int(config['batch_size_8']['batch_size'])
optimizer = eval(f"optim.{config['multilabel_DinoS_Parameters_2']['optimizer']}")
lr=float(config['multilabel_DinoS_Parameters_2']['lr'])
patience = int(config['multilabel_DinoS_Parameters_2']['patience'])
step_size = int(config['multilabel_DinoS_Parameters_2']['step_size'])
gamma = float(config['multilabel_DinoS_Parameters_2']['gamma'])

N_QUERIES = 13
POWER = 1
FILENAME = "AL_max_score_results_for_multilabel_classification.pickle"
SPECIFIC_PATH = 'ActiveLearning/Multilabel/DinoS/max_score_seed45_config2'
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

train_df = train_df.sample(frac=1, random_state=1234)
initial_df = train_df[train_df['image'].isin(predefined_image_names)]
initial_df = initial_df.set_index('image').loc[predefined_image_names].reset_index()
missing_images = [img for img in predefined_image_names if img not in train_df['image'].values]
if missing_images:
    print("These images are missing from train_df:", missing_images)
    assert not missing_images, "Initial DataFrame doesn't contain the specified images."

pool_df = train_df.drop(initial_df.index)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

initial_dataset = CustomDataset(initial_df, image_dir, transform)
pool_dataset = CustomDataset(pool_df, image_dir, transform)
val_dataset = CustomDataset(val_df, image_dir, transform)
test_dataset = CustomDataset(test_df, image_dir, transform)
train_dataset = CustomDataset(train_df, image_dir, transform)

initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

class CustomDINONormModel(nn.Module):
    """
    A custom model that integrates a pre-trained DINO model with
    additional normalization and classification layers.

    Attributes:
        dino_model (nn.Module): The pre-trained DINO model.
        classifier (nn.Sequential): The classification layers added on top of the DINO model.
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
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the DINO model and the classifier.

        Raises:
            ValueError: If the input tensor dimensions are unexpected.
        """
        if x.dim() == 4 and x.shape[3] == 3:  # (N, H, W, C)
            x = x.permute(0, 3, 1, 2)  # (N, C, H, W)
        elif x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Unexpected input dimensions: {x.shape}")
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

dinoL_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
model = CustomDINONormModel(dinoL_model, num_classes=3).to(DEVICE)

def tensors_to_numpy(loader):
    """
    Converts a batch of tensors from a DataLoader to numpy arrays.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader containing batches of tensors.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - images (np.ndarray): Numpy array of images with shape (N, H, W, C).
            - labels (np.ndarray): Numpy array of labels.
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

X_initial_np, y_initial_np = tensors_to_numpy(initial_loader)
X_pool_np, y_pool_np = tensors_to_numpy(pool_loader)
X_val_np, y_val_np = tensors_to_numpy(val_loader)
X_test_np, y_test_np = tensors_to_numpy(test_loader)
X_train_np, y_train_np = tensors_to_numpy(train_loader)
print("Adjusted shapes after loading:")
print(f"X_initial_np: {X_initial_np.shape}, y_initial_np: {y_initial_np.shape}")
print(f"X_pool_np: {X_pool_np.shape}, y_pool_np: {y_pool_np.shape}")
print(f"X_test_np: {X_test_np.shape}, y_test_np: {y_test_np.shape}")
print(f"X_val_np: {X_val_np.shape}, y_val_np: {y_val_np.shape}")

f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)
es = EarlyStopping(monitor='valid_loss', patience=patience, lower_is_better=True)
cp = Checkpoint(dirname=os.path.join(save_dir, 'model_checkpoints'),
                monitor='valid_loss_best', f_params='best_model.pt')
valid_ds = Dataset(X_val_np, y_val_np)
train_split = predefined_split(valid_ds)

class CustomNeuralNetClassifier(NeuralNetClassifier):
    def predict_proba(self, X, **kwargs):
        """
        Returns probabilities in the shape (N, C) where:
        - N: Number of samples
        - C: Number of classes
        """
        # Utilize skorch's built-in predict_proba which applies sigmoid
        proba = super().predict_proba(X, **kwargs)  # Expected Shape: (N, C)
        #print(f"predict_proba shape: {proba.shape}")  # Debugging
        # Check if proba has unexpected dimensions
        if proba.ndim == 3 and proba.shape[1] == 2:
            # Assuming the second dimension corresponds to binary classes
            # Extract the probability for class '1' for each class
            proba = proba[:, 1, :]  # Shape: (N, C)
            #print(f"Adjusted predict_proba shape: {proba.shape}")  # Debugging
        return proba

    def predict(self, X, threshold=0.5, **kwargs):
        """
        Returns binary predictions in the shape (N, C) by applying a threshold
        to the probabilities of class '1'.

        Parameters:
        - threshold: Threshold for classifying as '1'. Default is 0.5.
        """
        proba = self.predict_proba(X, **kwargs)  # Expected Shape: (N, C)
        #print(f"predict shape: {proba.shape}")  # Debugging
        return (proba >= threshold).astype(int)
    
lr_scheduler = LRScheduler(
    policy=lr_scheduler.StepLR,
    step_size=step_size,
    gamma=gamma
)

classifier = CustomNeuralNetClassifier(
    module=model,
    criterion=criterion,
    batch_size=batch_size,
    optimizer=optimizer,
    lr=lr,
    max_epochs=100,
    train_split=train_split,
    device=DEVICE,
    callbacks=[train_f1, valid_f1, csv_logger, es, cp, lr_scheduler],
    verbose=1
)

learner = ActiveLearner(
    estimator=classifier,
    query_strategy=max_score,
    X_training=X_initial_np,
    y_training=y_initial_np
)

pre_f1 = f1_score(y_test_np, learner.predict(X_test_np), average='micro')
pre_f1_macro = f1_score(y_test_np, learner.predict(X_test_np), average='macro', zero_division=1)
pre_acc = learner.score(X_test_np, y_test_np)

print(f"Pre F1 micro score = {pre_f1:.4f}")
print(f"Pre F1 macro score = {pre_f1_macro:.4f}")
print(f"Pre Accuracy = {pre_acc:.4f}")

# Function to adjust batch size
def batch(number, base):
    """Returns the nearest batch size multiple for the provided number of samples."""
    multiple = base * round(number / base)
    return multiple

acc_test_data = []
f1_test_data = []
performance_val_data = []
performance_val_data_with_loss = []
f1_test_data_macro = []
performance_val_data_macro = []
no_of_samples = []
total_samples = X_initial_np.shape[0]
performance_val_data_with_loss.append(learner.estimator.history_)

f1_test_data.append(pre_f1)
f1_test_data_macro.append(pre_f1_macro)
acc_test_data.append(pre_acc)

samples_log_file = os.path.join(save_dir, "sample_selection_log.csv")
with open(samples_log_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Sample_Names"])

initial_image_list = initial_df["image"].tolist()
with open(samples_log_file, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([1] + initial_image_list)

for i in range(N_QUERIES):
    print(f"\nIteration: {i + 1}")

    gc.collect()
    torch.cuda.empty_cache()

    if i != 12:
        POWER += 0.25
        n_instances = batch(int(np.ceil(np.power(10, POWER))), batch_size)
        print(f"Selecting {n_instances} informative samples: ")
        query_index, _ = learner.query(X_pool_np, n_instances=n_instances)
        query_index = np.unique(query_index)

        total_samples = total_samples + n_instances
        print(f"\nTraining started with {total_samples} samples:\n")

        selected_sample_names = train_df.loc[query_index, "image"].tolist()
        with open(samples_log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([i + 2] + selected_sample_names)

        learner.teach(X=X_pool_np[query_index], y=y_pool_np[query_index])
    else:
        n_instances = X_pool_np.shape[0]
        print(f"Selecting {n_instances} informative samples: ")

        query_index, _ = learner.query(X_pool_np, n_instances=n_instances)
        query_index = np.unique(query_index)

        total_samples = total_samples + n_instances
        print(f"\nTraining started with {total_samples} samples:\n")

        selected_sample_names = train_df.loc[query_index, "image"].tolist()
        with open(samples_log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([i + 2] + selected_sample_names)

        learner.teach(X=X_pool_np[query_index], y=y_pool_np[query_index])

    y_pred_val = learner.predict(X_val_np)
    val_accuracy = accuracy_score(y_val_np, y_pred_val)
    val_f1 = f1_score(y_val_np, y_pred_val, average='micro')
    val_f1_macro = f1_score(y_val_np, y_pred_val, average='macro')
    performance_val_data.append({'accuracy': val_accuracy, 'f1_micro': val_f1})
    performance_val_data_macro.append({'f1_macro': val_f1_macro})
    performance_val_data_with_loss.append(learner.estimator.history_)

    no_of_samples.append(total_samples)
    print(no_of_samples)

    y_pred = learner.predict(X_test_np)
    f1 = f1_score(y_test_np, y_pred, average='micro')
    f1_macro = f1_score(y_test_np, y_pred, average='macro')
    f1_test_data.append(f1)
    f1_test_data_macro.append(f1_macro)
    print(f"F1 Micro Score after query {i + 1}: {f1}")
    print(f"F1 Macro Score after query {i + 1}: {f1_macro}")
    print(f"Number of samples used for retraining: {total_samples}")

    X_pool_np = np.delete(X_pool_np, query_index, axis=0)
    y_pool_np = np.delete(y_pool_np, query_index, axis=0)
    print(f"Number of samples in pool after training and deleting samples: {X_pool_np.shape[0]}")

    checkpoint_path = os.path.join(save_dir, f"model_checkpoint_iteration_{i}.pt")
    torch.save(classifier.module_.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    best_model_path = os.path.join(save_dir, 'model_checkpoints', 'best_model.pt')

    classifier.load_params(best_model_path)
    gc.collect()
    torch.cuda.empty_cache()

data_dict = {"test_f1_scores_micro": f1_test_data,
    "test_f1_scores_macro": f1_test_data_macro,
    "val_performance_with_loss":performance_val_data_with_loss,
    "val_performance_micro": performance_val_data,
    "val_performance_macro": performance_val_data_macro,
    "test_accuracy": acc_test_data,
    "no_of_samples": no_of_samples
    }

file_path = os.path.join(save_dir, FILENAME)

with open(file_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file, protocol = pickle.HIGHEST_PROTOCOL)

print(f"Pickle file saved to {file_path}")
