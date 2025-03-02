"""
This module implements Multiclass Active Learning using the Margin Sampling strategy.

The primary purpose of this module is to facilitate the active learning
process for multilabel classification tasks.
It includes functions and classes to load data,
train models, and evaluate performance using the average confidence
strategy to select the most informative samples for labeling.
"""
import os
import gc
import socket
import configparser
import pickle
import random
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from skorch.callbacks import EpochScoring
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch.classifier import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, Checkpoint, Callback, LRScheduler
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from torch.utils.data import DataLoader
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling


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

seed = int(config['seed_42']['seed'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

predefined_image_names = config['predefined_images']['image_names'].split(', ')

batch_size = int(config['multiclass_EffNet_Parameters_2']['batch_size'])
dropout = float(config['multiclass_EffNet_Parameters_2']['dropout'])
layer_freeze = config['multiclass_EffNet_Parameters_2']['layer_freeze']
criterion = eval(f"nn.{config['multiclass_EffNet_Parameters_2']['criterion']}")
optimizer = eval(f"optim.{config['multiclass_EffNet_Parameters_2']['optimizer']}")
lr = float(config['multiclass_EffNet_Parameters_2']['lr'])
patience = int(config['multiclass_EffNet_Parameters_2']['patience'])
step_size = int(config['multiclass_EffNet_Parameters_2']['step_size'])
gamma = float(config['multiclass_EffNet_Parameters_2']['gamma'])
#fc_units = int(config['multiclass_EffNet_Parameters_2']['fc_units'])

POWER = 1
N_QUERIES = 13
NUM_CLASSES = 8
FILENAME = "AL_margin_sampling_results_for_multiclass_classification_s42.pickle"
SPECIFIC_PATH = 'ActiveLearning/Multiclass/EfficientNet/uncertainty_sampling_seed42_param2'
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

multiclass_train_df = train_df.assign(multiclass=multiclass_labels[0])
multiclass_train_df = multiclass_train_df[['image', 'multiclass']].dropna()

multiclass_test_df = test_df.assign(multiclass=multiclass_labels[1])
multiclass_test_df = multiclass_test_df[['image', 'multiclass']].dropna()

multiclass_val_df = val_df.assign(multiclass=multiclass_labels[2])
multiclass_val_df = multiclass_val_df[['image', 'multiclass']].dropna()

print(f"multiclass_train_df shape: {multiclass_train_df.shape}")
print(f"multiclass_test_df shape: {multiclass_test_df.shape}")
print(f"multiclass_val_df shape: {multiclass_val_df.shape}")


def batch(number, base):
    """Returns the nearest batch size multiple for the provided number of samples."""
    multiple = base * round(np.divide(number, base))
    return multiple


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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

initial_dataset = CustomDataset(initial_df, image_dir, transform)
train_dataset = CustomDataset(multiclass_train_df, image_dir, transform)
pool_dataset = CustomDataset(pool_df, image_dir, transform)
val_dataset = CustomDataset(multiclass_val_df, image_dir, transform)
test_dataset = CustomDataset(multiclass_test_df, image_dir, transform)

initial_loader = DataLoader(
    initial_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CSVLogger(Callback):
    """Log epoch data to a CSV file."""

    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        self.file_exist = os.path.exists(filename)

    def on_epoch_end(self, net, **kwargs):
        logs = {key: net.history[-1, key] for key in self.fieldnames}
        if not self.file_exist:
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                csv_writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                csv_writer.writeheader()
            self.file_exist = True
        with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
            csv_writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            csv_writer.writerow(logs)


save_dir = os.path.join(base_dir, SPECIFIC_PATH)
os.makedirs(save_dir, exist_ok=True)

csv_logger = CSVLogger(
    os.path.join(save_dir, "training_history.csv"),
    ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']
)


'''def define_model(layer_freeze_upto, fc_units, dropout_rate, num_classes):
    """
    Defines and returns an EfficientNetV2-L model with specified layers frozen and a custom classifier.

    Args:
        layer_freeze_upto (str): The name of the layer up to which parameters should be frozen.
        fc_units (int): The number of units in the fully connected layer.
        dropout_rate (float): The dropout rate for the dropout layer.
        num_classes (int): The number of output classes for the classifier.

    Returns:
        nn.Module: The modified EfficientNetV2-L model with the custom classifier.
    """

    model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

    cutoff_reached = False
    for name, param in model.named_parameters():
        if not cutoff_reached:
            if name == layer_freeze_upto:
                cutoff_reached = True
            param.requires_grad = False
        else:
            param.requires_grad = True

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, fc_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(fc_units, num_classes),
    )

    return model'''


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

learner = ActiveLearner(
    estimator=classifier,
    query_strategy=uncertainty_sampling,
    X_training=X_initial_np,
    y_training=y_initial_np
)

pre_f1 = f1_score(y_test_np, learner.predict(X_test_np), average='micro')
pre_f1_macro = f1_score(y_test_np, learner.predict(X_test_np), average='macro')

print(f"Pre F1 micro score = {pre_f1:.4f}")
print(f"Pre F1 macro score = {pre_f1_macro:.4f}")

f1_test_data = []
performance_val_data = []
performance_val_data_with_loss = []
f1_test_data_macro = []
performance_val_data_macro = []
no_of_samples = []
total_samples = X_initial_np.shape[0]
no_of_samples.append(total_samples)

performance_val_data_with_loss.append(learner.estimator.history_)
f1_test_data.append(pre_f1)
f1_test_data_macro.append(pre_f1_macro)

samples_log_file = os.path.join(save_dir, "sample_selection_log.csv")
with open(samples_log_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Sample_Names"])

initial_image_list = initial_df["image"].tolist()
with open(samples_log_file, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([1] + initial_image_list)

for i in range(N_QUERIES):
    print("\nIteration: ", i + 1)
    gc.collect()
    torch.cuda.empty_cache()
    if i != 12:
        POWER += 0.25
        n_instances = batch(int(np.ceil(np.power(10, POWER))), batch_size)

        print(f"Selecting {n_instances} informative samples: ")

        query_index, query_instance = learner.query(
            X_pool_np, n_instances=n_instances)

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

        query_index, query_instance = learner.query(
            X_pool_np, n_instances=n_instances)

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
    print(
        f"Number of samples in pool after training and deleting samples: {X_pool_np.shape[0]}")

    checkpoint_path = os.path.join(
        save_dir, f"model_checkpoint_iteration_{i}.pt")
    torch.save(classifier.module_.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    best_model_path = os.path.join(
        save_dir, 'model_checkpoints', 'best_model.pt')

    classifier.load_params(best_model_path)
    gc.collect()
    torch.cuda.empty_cache()

data_dict = {"test_f1_scores_micro": f1_test_data,
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