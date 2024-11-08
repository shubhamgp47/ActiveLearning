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

# Define image directory and load dataframes
image_dir = os.path.abspath('D:/linear_winding_images_with_labels/')
df_dir = os.path.abspath('D:/datasets/')
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
from modAL.uncertainty import uncertainty_sampling

query_strategy = uncertainty_sampling

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

'''class CustomDataset(Dataset):
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
        return image, label'''

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

if __name__ == "__main__":
    '''# Function to prepare data
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
        return images, labels'''

    '''# Prepare initial data
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
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")'''

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
    #save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/multiclass/FixSampling/MyNewParameters/DinoSmall/{run_id}"
    save_dir = os.path.abspath('D:/Shubham/results/US/Fixuncertainty_sampling')
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

    #Hyperparameters
    num_classes = 8
    initial_samples = 8

    from skorch.callbacks import EarlyStopping, Checkpoint, Callback

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


    # Function to convert one-hot encoded labels to integer labels
    def convert_one_hot_to_labels(y):
        return np.argmax(y, axis=1) if len(y.shape) > 1 else y

    '''# Ensure that y_train_initial, y_pool_initial, y_val, and y_test are in the correct format
    y_train_initial_np = convert_one_hot_to_labels(y_train_initial.clone().detach().cpu().numpy())
    y_pool_initial_np = convert_one_hot_to_labels(y_oracle.clone().detach().cpu().numpy())
    y_val_np = convert_one_hot_to_labels(y_val.clone().detach().cpu().numpy())
    y_test_np = convert_one_hot_to_labels(y_test.clone().detach().cpu().numpy())

    # Convert initial datasets to NumPy
    X_train_initial_np = X_train_initial.clone().detach().cpu().numpy()
    X_pool_np = X_pool.clone().detach().cpu().numpy()
    X_val_np = X_val.clone().detach().cpu().numpy()
    X_test_np = X_test.clone().detach().cpu().numpy()'''

    # Check shapes
    print("Adjusted shapes after loading:")
    print("X_initial_np:", X_initial_np.shape, "y_initial_np:", y_initial_np.shape)
    print("X_train_np:", X_train_np.shape, "y_train_np:", y_train_np.shape)
    print("X_pool_np:", X_pool_np.shape, "y_pool_np:", y_pool_np.shape)
    print("X_test_np:", X_test_np.shape, "y_test_np:", y_test_np.shape)
    print("X_val_np:", X_val_np.shape, "y_val_np:", y_val_np.shape)

    # Initialize cumulative datasets
    X_cumulative = X_initial_np
    y_cumulative = y_initial_np
    '''X_cumulative = X_train_np[:initial_samples]
    y_cumulative = y_train_np[:initial_samples]'''

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




    '''classifier = NeuralNetClassifier(
        module=model,
        criterion=nn.CrossEntropyLoss(),
        #optimizer=optim.RMSprop,
        optimizer = optim.Adam,
        lr=0.0000029836174096485545,
        max_epochs=100,
        train_split=predefined_split(valid_ds),  # Use predefined split for validation
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

    # Initialize the ActiveLearner
    learner = ActiveLearner(
        estimator=classifier,
        query_strategy=query_strategy,
        X_training=X_cumulative,
        y_training=y_cumulative
    )



    # Initialize EarlyStopping and other callbacks
    total_samples = X_train_np.shape[0]

    # Active learning loop parameters
    n_queries = 14
    patience = 19 # Number of rounds to continue without improvement
    wait = 0  # Current wait time
    best_f1_score = 0.0
    acc_test_data = []
    f1_test_data = []
    performance_val_data = []  
    no_of_samples = []  
    POWER = 1  # Initialize power for sample selection

    # File to save selected sample names per iteration
    samples_log_file = os.path.join(save_dir, "sample_selection_log.csv")
    with open(samples_log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Sample Names"])

    cumulative_image_names = initial_df["image"].tolist()
    for i in range(n_queries):
        if i == 0:
            n_instances = initial_samples

            print(f"\nQuery {i + 1}: Using the initial {n_instances} samples already in X_cumulative and y_cumulative.")
            print(f"Training begins with initial {n_instances} samples.")

            # Log the initially selected sample names (for first iteration)
            '''initial_sample_names = train_df.loc[:initial_samples-1, "image"].tolist()  # Assuming first n samples are already present
            print(f"Selected samples in Query {i + 1}: {initial_sample_names}")'''
            initial_image_list = initial_df["image"].tolist()
            with open(samples_log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i + 1] + initial_image_list)
        elif i == 13:
            n_instances = X_pool_np.shape[0]
            print(f"\nQuery {i + 1}: Requesting {n_instances} samples.")
            print(f"Number of samples in pool before query: {X_pool_np.shape[0]}")
        
        else:
            POWER += 0.25
            n_instances = batch(int(np.ceil(np.power(10, POWER))), batch_size)
            print(f"\nQuery {i + 1}: Requesting {n_instances} samples.")
            print(f"Number of samples in pool before query: {X_pool_np.shape[0]}")
        

        if i != 0:
            query_idx, query_instance = learner.query(X_pool_np, n_instances=n_instances)
            X_query, y_query = X_pool_np[query_idx], y_pool_np[query_idx]
            y_query = convert_one_hot_to_labels(y_query)

            # Update cumulative datasets
            X_cumulative = np.vstack((X_cumulative, X_query))
            y_cumulative = np.concatenate((y_cumulative, y_query))

            # Log the selected sample names
            selected_sample_names = train_df.loc[query_idx, "image"].tolist()
            #print(f"Selected samples in Query {i + 1}: {selected_sample_names}")
            with open(samples_log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i + 1] + selected_sample_names)
            '''cumulative_sample_names = train_df.loc[train_df.index.isin(np.arange(len(X_cumulative))), "image"].tolist()
            with open(samples_log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i + 1] + cumulative_sample_names)'''
            #Use this for all cumulative images
            '''new_sample_names = train_df.iloc[query_idx, 0].tolist()  # Assuming this fetches the correct column for image names
            # Add new samples to the cumulative list (to keep track of all images)
            cumulative_image_names.extend(new_sample_names)'''
            
        
        # Retrain the learner with the cumulative data
        learner.teach(X=X_cumulative, y=y_cumulative, only_new=False)
        #Use this for all cumulative images
        '''with open(samples_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i + 1] + cumulative_image_names) '''                                                                                         

        # Evaluate learner's performance
        y_pred_val = learner.predict(X_val_np)
        val_accuracy = accuracy_score(y_val_np, y_pred_val)
        val_f1 = f1_score(y_val_np, y_pred_val, average='micro')
        performance_val_data.append({'accuracy': val_accuracy, 'f1': val_f1})

        no_of_samples.append(len(X_cumulative))

        y_pred = learner.predict(X_test_np)
        accuracy = accuracy_score(y_test_np, y_pred)
        f1 = f1_score(y_test_np, y_pred, average='micro')
        acc_test_data.append(accuracy)
        f1_test_data.append(f1)
        print(f"Accuracy after query {i + 1}: {accuracy}")
        print(f"F1 Score after query {i + 1}: {f1}")
        print(f"Number of samples used for retraining: {len(X_cumulative)}")

        # Early Stopping Check
        if f1 > best_f1_score:
            best_f1_score = f1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Stopping early due to no improvement in F1 score.")
                break
        
        if i != 0:
            # Remove queried instances from the pool
            X_pool_np = np.delete(X_pool_np, query_idx, axis=0)
            y_pool_np = np.delete(y_pool_np, query_idx, axis=0)
            print(f"Number of samples in pool after query: {X_pool_np.shape[0]}")

        checkpoint_path = f"D:/Shubham/results/US/Fixuncertainty_sampling/model_checkpoint_iteration_{i}.pt"
        torch.save(classifier.module_.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    # Log final performance
    print(f"Final F1 scores across iterations: {f1_test_data}")
    print(f"Final accuracies across iterations: {acc_test_data}")

    # Save performance results
    performance_filename = "D:/Shubham/results/US/Fixuncertainty_sampling/performance_results.npy"
    np.save(performance_filename, {
        "f1_scores": f1_test_data, 
        "accuracies": acc_test_data,
        "val_performance": performance_val_data,
        "no_of_samples": no_of_samples
    })
    print(f"Performance results saved to {performance_filename}")