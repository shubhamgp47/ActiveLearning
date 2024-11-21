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
from modAL.multilabel import avg_confidence
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import time
import csv
import plotly.graph_objects as go
import plotly.subplots as subplots
import random

seed_value = 42
#random_state = np.random.RandomState(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

image_dir = os.path.abspath('D:/linear_winding_images_with_labels/')
df_dir = os.path.abspath('D:/datasets/')
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")

# Path to the CSV file that contains the exact images to use for each iteration
#SELECTED_IMAGES_CSV_PATH = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/multilabel/imagesfrommulticlass/FixedsampleSelection/")
#SELECTED_IMAGES_CSV_PATH = os.path.abspath("C:/Users/localuserSG/ActiveLearning/MultiLabel01/DinoSmall/imagesFroMullticlass/Avg_confidence/")  
SELECTED_IMAGES_CSV_PATH = os.path.abspath("C:/Users/localuserSG/ActiveLearning/Multilabel01/DinoSmall/imagesFromMulticlass/Avg_confidence") 
selected_images_df = pd.read_csv(SELECTED_IMAGES_CSV_PATH + "/sample_selection_log_seed_resultof_MLParams.csv")

def batch(number, base):
    return base * round(number / base)


'''# Plotting functions
def plot_validation_performance_vs_epochs(performance, no_of_samples, title, x_axes, y_axes, plotting_curve):
    if type(performance) != list or type(no_of_samples) != list:
        raise TypeError("Input must be a list")
    
    plot = subplots.make_subplots(specs=[[{"secondary_y": False}]])
    plot.update_layout(plot_bgcolor='rgb(209, 217, 222)')
    
    if "loss" in plotting_curve:
        for i in range(len(performance)):
            plot.add_trace(
                go.Scatter(y=[data["val_loss"] for data in performance], name=f"{no_of_samples[i]} Images used: Val loss"),
                secondary_y=False,
            )
    elif "f1" in plotting_curve:
        for i in range(len(performance)):
            plot.add_trace(
                go.Scatter(y=[data["val_f1_score"] for data in performance], name=f"{no_of_samples[i]} Images used: val F1 score"),
                secondary_y=False,
            )
    elif "accuracy" in plotting_curve:  # Handling accuracy plotting
        for i, accuracy in enumerate(performance):
            plot.add_trace(
                go.Scatter(x=[no_of_samples[i]], y=[accuracy], name=f"Accuracy after query {i+1}: {accuracy:.3f}", mode='lines+markers'),
                secondary_y=False,
            )
    else:
        raise NameError("Unknown name given for plotting curve")
    
    plot.update_layout(
        title_text=str(title),
        legend=dict(font=dict(color='black')),
        title_font_color='black'
    )
    
    plot.update_xaxes(
        title_text=str(x_axes),
        title_font=dict(color="black", family="Arial"),
        tickfont_color='black'
    )
    
    plot.update_yaxes(
        title_text=str(y_axes),
        title_font=dict(color="black", family="Arial"),
        tickfont_color='black',
        secondary_y=False
    )
    
    plot.write_image(f"{str(title).replace(' ', '_')}.png")
    plot.show()

def plot_performance_vs_amount_of_data(performance, no_of_samples, title, x_axes, y_axes, y_axes_secondary=None, mode='validation'):
    plot = subplots.make_subplots(specs=[[{"secondary_y": y_axes_secondary is not None}]])
    plot.update_layout(plot_bgcolor='rgb(209, 217, 222)')
    
    # Add vertical lines and labels at sample points
    samples_text = [str(f) for f in no_of_samples]
    for line, label in zip(no_of_samples, samples_text):
        plot.add_trace(go.Scatter(
            x=[line, line],
            y=[0, 1.12],
            mode='lines',
            line=dict(dash='dash', color='rgb(149, 162, 171)'),
            showlegend=False,
            yaxis='y2'
        ))
        plot.add_trace(go.Scatter(
            x=[line],
            y=[1.13],
            mode='text',
            marker=dict(size=0),
            text=[label],
            textposition='top center',
            showlegend=False,
            yaxis='y2',
            textfont=dict(family='Arial', color='black', size=8.5)
        ))

    # Decide what to plot based on the mode
    if mode == "validation":
        if all(isinstance(item, dict) for item in performance):
            loss = [data["val_loss"][-1] for data in performance]
            f1 = [data["val_f1_score"][-1] for data in performance]
            plot.add_trace(
                go.Scatter(x=no_of_samples, y=loss, name="Validation Loss", marker=dict(color='rgb(97,192,134)')),
                secondary_y=False,
            )
            plot.add_trace(
                go.Scatter(x=no_of_samples, y=f1, name="Validation F1 Score", marker=dict(color='rgb(245,130,31)')),
                secondary_y=True,
            )
    elif mode == "testing":
        plot.add_trace(
            go.Scatter(x=no_of_samples, y=performance, name="Test F1 Score", marker=dict(color='rgb(151,193,57)')),
            secondary_y=False,
        )
    else:
        raise NameError(f"Unknown mode: {mode}")

    # Layout and axis updates
    plot.update_layout(
        title_text=title,
        legend=dict(font=dict(color='black')),
        title_font_color='black'
    )
    plot.update_xaxes(
        title_text=x_axes,
        title_font=dict(color="black", family="Arial"),
        tickfont_color='black',
        type="log"
    )
    plot.update_yaxes(
        title_text=y_axes,
        title_font=dict(color="black", family="Arial"),
        tickfont_color='black',
        secondary_y=False
    )
    if y_axes_secondary:
        plot.update_yaxes(
            title_text=y_axes_secondary,
            title_font=dict(color="black", family="Arial"),
            tickfont_color='black',
            secondary_y=True
        )

    plot.write_image(f"{title.replace(' ', '_')}.png")
    plot.show()
'''


# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

'''# Function to load images
def load_images(image_dir, df):
    images, labels = [], []
    for i in tqdm(range(df.shape[0])):
        image_path = os.path.join(image_dir, df.loc[i, "image"])
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        images.append(img)
        labels.append(df.drop(columns=['image', 'binary_NOK']).iloc[i].values.astype('float32'))
    return torch.stack(images), torch.tensor(labels)'''

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
#save_dir = os.path.abspath('D:/Shubham/results/multilabel01/DinoSmall/ActiveLearning/imagesFroMullticlass/withseed/avg_confidence')
save_dir = os.path.abspath('D:/Shubham/results/multilabel01/DinoSmall/experiment/AL/')
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

'''# Check shapes
print("Adjusted shapes after loading:")
print("X_train_np:", X_train_np.shape, "y_train_np:", y_train_np.shape)
print("X_pool:", X_pool.shape, "y_pool:", y_pool.shape)
print("X_test_np:", X_test_np.shape, "y_test_np:", y_test_np.shape)
print("X_val_np:", X_val_np.shape, "y_val_np:", y_val_np.shape)'''

# Skorch callbacks and classifier setup
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)
es = EarlyStopping(monitor='valid_loss', patience=13, lower_is_better=True)
model_checkpoint = Checkpoint(dirname=os.path.join(save_dir, 'model_checkpoints'), monitor='valid_loss_best')

'''def get_images_and_labels_for_iteration(selected_images_df, iteration, image_dir, label_df):
    # Get the selected images for this iteration as a single string
    images_for_iteration = selected_images_df[selected_images_df["Query_Iteration"] == iteration]["Selected_Image"].values[0]
    
    # Split the images string by semicolon to get individual filenames
    image_list = images_for_iteration.split(';')

    loaded_images = []
    labels = []

    for img_name in image_list:
        img_name = img_name.strip()  # Remove any leading/trailing whitespace
        image_path = os.path.join(image_dir, img_name)
        
        # Load and transform the image
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        loaded_images.append(img)

        # Fetch the corresponding labels from the label_df based on the image name
        label = label_df[label_df["image"] == img_name][['multi-label_double_winding', 'multi-label_gap', 'multi-label_crossing']].values
        labels.append(label)
        labels = np.array(labels)
    
    # Convert lists to torch tensors and return
    return torch.stack(loaded_images), torch.tensor(labels).squeeze().float()'''

def get_images_and_labels_for_iteration(selected_images_df, iteration, image_dir, label_df):
    # Get the selected images for this iteration as a single string
    images_for_iteration = selected_images_df[selected_images_df["Iteration"] == iteration]["Sample_Names"].values[0]
    
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

# Define the active learner
learner = ActiveLearner(
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
    ),
    query_strategy=avg_confidence,
    X_training=X_query_initial.numpy(),
    y_training=y_query_initial.numpy()
)

pre_f1 = f1_score(y_test_np, learner.predict(X_test_np), average='macro')
pre_acc = learner.score(X_test_np, y_test_np)

print(f"Pre F1 score = {pre_f1:.3f}")

n_queries = 13
best_f1_score = 0
wait = 0
patience = 13
acc_test_data = []
f1_test_data = []

f1_test_data.append(pre_f1)

'''X_cumulative = X_query_initial
y_cumulative = y_query_initial'''



# Active Learning Loop with Corrected Label Handling
for i in range(n_queries):
    print(f"\nQuery {i + 1}: Using the exact images and labels from the CSV for this iteration.")

    # Load the exact images and their corresponding labels for this query iteration
    X_query, y_query = get_images_and_labels_for_iteration(selected_images_df, iteration=i + 2, image_dir=image_dir, label_df=train_df)

    # Convert to numpy
    X_query_np = X_query.numpy()
    y_query_np = y_query.numpy()

    #X_cumulative = np.vstack((X_cumulative, X_query_np))
    #y_cumulative = np.vstack((y_cumulative, y_query_np))  # Use vstack to match dimensions
    #print(f"Number of samples used for training in Query {i + 1} is {len(X_cumulative)}")
    print(f"Number of new samples used for training in Query {i + 1} is {len(X_query_np)}")
    #To Do - add total samples logic here.

    # Teach the learner with the new data
    learner.teach(X=X_query_np , y=y_query_np)

    # Evaluate on test set pre_f1 = f1_score(y_test_np, learner.predict(X_test_np), average='macro')
    y_pred = learner.predict(X_test_np)
    y_pred = torch.sigmoid(torch.tensor(y_pred)) > 0.5
    accuracy = accuracy_score(y_test_np, y_pred.numpy())
    f1 = f1_score(y_test_np, y_pred.numpy(), average='macro')
    acc_test_data.append(accuracy)
    f1_test_data.append(f1)

    print(f"Accuracy after query {i + 1}: {accuracy}")
    print(f"F1 Score after query {i + 1}: {f1}")

    if f1 > best_f1_score:
        best_f1_score = f1

    checkpoint_path = os.path.join(save_dir, f"model_checkpoint_iteration_{i}.pt")
    torch.save(learner.estimator.module_.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

print(f"Best F1 score across iterations: {best_f1_score}")
performance_filename = os.path.join(save_dir, "performance_results.npy")
np.save(performance_filename, {
    "f1_scores": f1_test_data
})
print(f"Performance results saved to {performance_filename}")