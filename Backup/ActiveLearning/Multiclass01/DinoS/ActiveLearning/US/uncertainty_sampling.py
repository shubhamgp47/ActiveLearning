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
from skorch.callbacks import EpochScoring
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch.classifier import NeuralNetClassifier
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from skorch.callbacks import EarlyStopping, Checkpoint, Callback
import torch.optim as optim
import csv

torch.multiprocessing.set_sharing_strategy('file_system')

# Define image directory and load dataframes
image_dir = os.path.abspath('D:/linear_winding_images_with_labels/')
df_dir = os.path.abspath('D:/datasets/')
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)

# Define specific image names to include in the initial dataset
predefined_image_names = [
    "Spule035_Image0269.jpg", "Spule020_Image0191.jpg", "Spule030_Image0310.jpg",
    "Spule013_Image0317.jpg", "Spule006_Image0201.jpg", "Spule020_Image0292.jpg",
    "Spule012_Image0854.jpg", "Spule020_Image0498.jpg"
]

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

# MultiClass training, test, and validation data frames
multiclass_train_df = train_df.assign(multiclass=multiclass_labels[0])
multiclass_train_df = multiclass_train_df[['image', 'multiclass']].dropna()

multiclass_test_df = test_df.assign(multiclass=multiclass_labels[1])
multiclass_test_df = multiclass_test_df[['image', 'multiclass']].dropna()

multiclass_val_df = val_df.assign(multiclass=multiclass_labels[2])
multiclass_val_df = multiclass_val_df[['image', 'multiclass']].dropna()

print(f"multiclass_train_df shape: {multiclass_train_df.shape}")
print(f"multiclass_test_df shape: {multiclass_test_df.shape}")
print(f"multiclass_val_df shape: {multiclass_val_df.shape}")

batch_size = 8
query_strategy = uncertainty_sampling
def batch(number, base):
    """Returns the nearest batch size multiple for the provided number of samples."""
    multiple = base * round(np.divide(number, base))
    return multiple

power = 1
train_size = batch(int(np.ceil(np.power(10, power))), batch_size)
print(f"Calculated train_size: {train_size}")

# Shuffle the dataframe
multiclass_train_df = multiclass_train_df.sample(frac=1, random_state=1234)

# Filter multiclass_train_df for initial images
initial_df = multiclass_train_df[multiclass_train_df['image'].isin(predefined_image_names)]
initial_df = initial_df.set_index('image').loc[predefined_image_names].reset_index()

missing_images = [img for img in predefined_image_names if img not in multiclass_train_df['image'].values]
if missing_images:
    print("These images are missing from multiclass_train_df:", missing_images)
    assert not missing_images, "Initial DataFrame doesn't contain the specified image."

pool_df = multiclass_train_df.drop(initial_df.index)

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

initial_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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

save_dir = os.path.abspath('D:/Shubham/results/Multiclass01/DinoSmall/ActiveLearning/uncertainty_sampling/')
os.makedirs(save_dir, exist_ok=True)

csv_logger = CSVLogger(
    os.path.join(save_dir, "training_history.csv"),
    ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']
)

class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, num_classes):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.35659850739606247),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.dino_model(x)
        x = self.classifier(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
for param in list(dino_model.parameters())[:65]:
    param.requires_grad = False

model = CustomDINONormModel(dino_model, num_classes=8).to(device)

num_classes = 8
initial_samples = 8

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

# Scoring metrics
f1_scorer = make_scorer(f1_score, average='micro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)

# Validation split and Early stopping
valid_ds = Dataset(X_val_np, y_val_np)
train_split = predefined_split(valid_ds)
es = EarlyStopping(monitor='valid_loss', patience=15, lower_is_better=True)
cp = Checkpoint(dirname='model_checkpoints', monitor='valid_loss_best')

'''classifier = NeuralNetClassifier(
    module=model,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.SGD,
    optimizer__momentum=0.24729309193472406,
    lr=0.002,
    max_epochs=100,
    train_split=train_split,
    device=device,
    callbacks=[train_f1, valid_f1, es, cp, csv_logger],
    verbose=1
)'''
classifier = NeuralNetClassifier(
        module=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.RMSprop,
        lr=0.000002,
        max_epochs=100,
        train_split=train_split,
        device=device,
        callbacks=[train_f1, valid_f1, es, cp, csv_logger],
        verbose=1
    )

learner = ActiveLearner(
    estimator=classifier,
    query_strategy=query_strategy,
    X_training=X_initial_np,
    y_training=y_initial_np
)

pre_f1 = f1_score(y_test_np, learner.predict(X_test_np), average='micro')
pre_acc = learner.score(X_test_np, y_test_np)

print(f"Pre F1 score = {pre_f1:.3f}")

# Active learning loop parameters
n_queries = 13
patience = 8
wait = 0
best_f1_score = 0.0
acc_test_data = []
f1_test_data = []
performance_val_data = []
no_of_samples = []
POWER = 1
# Initialize EarlyStopping and other callbacks
total_samples = X_initial_np.shape[0]


acc_test_data.append(pre_acc)
f1_test_data.append(pre_f1)

# File to save selected sample names per iteration
samples_log_file = os.path.join(save_dir, "sample_selection_log.csv")
with open(samples_log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Sample_Names"])

initial_image_list = initial_df["image"].tolist()
with open(samples_log_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([1] + initial_image_list)

# Active learning loop
for i in range(n_queries):
    print("\nIteration: ", i + 1)

    if i != 12:
        POWER += 0.25
        n_instances = batch(int(np.ceil(np.power(10, POWER))), batch_size)

        print(f"Selecting {n_instances} informative samples: ")

        query_index, query_instance = learner.query(X_pool_np, n_instances=n_instances)

        total_samples = total_samples + n_instances
        print(f"\nTraining started with {total_samples} samples:\n")

        selected_sample_names = train_df.loc[query_index, "image"].tolist()
        with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 2] + selected_sample_names)

        learner.teach(X=X_pool_np[query_index], y=y_pool_np[query_index])

    else:
        n_instances = X_pool_np.shape[0]
        print(f"Selecting {n_instances} informative samples: ")

        query_index, query_instance = learner.query(X_pool_np, n_instances=n_instances)

        total_samples = total_samples + n_instances
        print(f"\nTraining started with {total_samples} samples:\n")

        selected_sample_names = train_df.loc[query_index, "image"].tolist()
        with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 2] + selected_sample_names)

        learner.teach(X=X_pool_np[query_index], y=y_pool_np[query_index])

    y_pred_val = learner.predict(X_val_np)
    val_accuracy = accuracy_score(y_val_np, y_pred_val)
    val_f1 = f1_score(y_val_np, y_pred_val, average='micro')
    performance_val_data.append({'accuracy': val_accuracy, 'f1': val_f1})

    no_of_samples.append(total_samples)

    y_pred = learner.predict(X_test_np)
    f1 = f1_score(y_test_np, y_pred, average='micro')
    f1_test_data.append(f1)
    print(f"F1 Score after query {i + 1}: {f1}")
    print(f"Number of samples used for retraining: {total_samples}")

    X_pool_np = np.delete(X_pool_np, query_index, axis=0)
    y_pool_np = np.delete(y_pool_np, query_index, axis=0)
    print(f"Number of samples in pool after training and deleting samples: {X_pool_np.shape[0]}")

    checkpoint_path = os.path.join(save_dir, f"model_checkpoint_iteration_{i}.pt")
    torch.save(classifier.module_.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

# Save performance results
performance_filename = os.path.join(save_dir, "performance_results.npy")
np.save(performance_filename, {
    "f1_scores": f1_test_data,
    "accuracies": acc_test_data,
    "val_performance": performance_val_data,
    "no_of_samples": no_of_samples
})
print(f"Performance results saved to {performance_filename}")
