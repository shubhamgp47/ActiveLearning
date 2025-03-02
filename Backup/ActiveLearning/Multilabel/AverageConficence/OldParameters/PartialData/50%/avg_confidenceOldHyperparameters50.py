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

# Directory and CSV setup
image_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")
df_dir_25 = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/Splits_v2024-03-18/")
df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
train_df = pd.read_csv(df_dir_25 + "/train_v2024-03-18_50%.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")

def batch(number, base):
    return base * round(number / base)


# Plotting functions
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



# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load images
def load_images(image_dir, df):
    images, labels = [], []
    for i in tqdm(range(df.shape[0])):
        image_path = os.path.join(image_dir, df.loc[i, "image"])
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        images.append(img)
        labels.append(df.drop(columns=['image', 'binary_NOK']).iloc[i].values.astype('float32'))
    return torch.stack(images), torch.tensor(labels)

# Load datasets
X_train_initial, y_train_initial = load_images(image_dir, train_df)
X_val, y_val = load_images(image_dir, val_df)
X_test, y_test = load_images(image_dir, test_df)

# Function to create data loaders
def create_data_loader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Function to split datasets
def split_datasets(X, y, train_size):
    return random_split(TensorDataset(X, y), [train_size, X.size(0) - train_size])

# Adjust batch size
batch_size = 4
power = 1
train_size = int(np.ceil(np.power(10, power)))

# Split datasets
initial_dataset, pool_dataset = split_datasets(X_train_initial, y_train_initial, train_size)

# Create data loaders
train_loader = DataLoader(initial_dataset, batch_size=batch_size, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=True)
val_loader = create_data_loader(X_val, y_val, batch_size)
test_loader = create_data_loader(X_test, y_test, batch_size)

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
save_dir = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/MultiLabel/minimum_average/final/{run_id}"
os.makedirs(save_dir, exist_ok=True)

# Field names for the logger
fieldnames = ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']
csv_logger = CSVLogger(os.path.join(save_dir, "training_history.csv"), fieldnames)

# Set proxy if necessary
os.environ['http_proxy'] = 'http://proxy:80'
os.environ['https_proxy'] = 'http://proxy:80'

# Custom model class
class CustomDINONormModel(nn.Module):
    def __init__(self, dino_model, num_classes):
        super(CustomDINONormModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.0032957439464482152),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
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

dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
for param in list(dino_model.parameters())[:173]:
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

# Load and convert data
X_train_np, y_train_np = tensors_to_numpy(train_loader)
X_pool, y_pool = tensors_to_numpy(pool_loader)
X_test_np, y_test_np = tensors_to_numpy(test_loader)
X_val_np, y_val_np = tensors_to_numpy(val_loader)

# Check shapes
print("Adjusted shapes after loading:")
print("X_train_np:", X_train_np.shape, "y_train_np:", y_train_np.shape)
print("X_pool:", X_pool.shape, "y_pool:", y_pool.shape)
print("X_test_np:", X_test_np.shape, "y_test_np:", y_test_np.shape)
print("X_val_np:", X_val_np.shape, "y_val_np:", y_val_np.shape)

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
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=optim.SGD,
    lr=0.0000018947665630084095,
    max_epochs=100,
    train_split=predefined_split(valid_ds),
    device=device,
    callbacks=[train_f1, valid_f1, es, model_checkpoint],
    verbose=1
)'''

classifier = NeuralNetClassifier(
    module=model,
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=optim.RMSprop,
    lr=0.00009732702526660113,
    max_epochs=100,
    train_split=predefined_split(valid_ds),  # Use predefined split for validation
    device=device,
    callbacks=[train_f1, valid_f1, es, model_checkpoint],
    verbose=1
)

initial_samples = 8
X_train_initial_np = X_train_np[:initial_samples].shape
y_train_initial_np = y_train_np[:initial_samples].shape

# Convert initial samples to the correct shape
X_cumulative = np.copy(X_train_initial_np)
y_cumulative = np.copy(y_train_initial_np)

print("X_cumulative shape: ", X_cumulative.shape)
print("y_cumulative shape: ", y_cumulative.shape)

# Active Learning setup
learner = ActiveLearner(
    estimator=classifier,
    query_strategy=avg_confidence,
    X_training=X_train_np,
    y_training=y_train_np
)

# Function to save checkpoints
def save_checkpoint(iteration, X_data, y_data, fieldnames, save_dir):
    iteration_dir = os.path.join(save_dir, f"checkpoint_{iteration}")
    os.makedirs(iteration_dir, exist_ok=True)
    np.save(os.path.join(iteration_dir, "X_data.npy"), X_data)
    np.save(os.path.join(iteration_dir, "y_data.npy"), y_data)
    print(f"Saved data at iteration {iteration} to {iteration_dir}")

# Define initial variables for the active learning loop
n_queries = 13
best_f1_score = 0
wait = 0
patience = 20  # Number of epochs to wait for improvement
total_samples = len(initial_dataset)
power = 1
acc_test_data = []
f1_test_data = []

# Active learning loop
for i in range(n_queries):
    if i == 12:  # Since iteration 13 means the 12th index (0-based index)
        n_instances = X_pool.shape[0]
    else:
       n_instances = batch(int(np.ceil(np.power(10, power))), batch_size)

    print(f"\nQuery {i + 1}: Requesting {n_instances} samples from pool of size {X_pool.shape[0]}")

    if X_pool.shape[0] < n_instances:
        print("Not enough samples left in the pool to query the desired number of instances.")
        break

    query_idx, _ = learner.query(X_pool, n_instances=n_instances)
    query_idx = np.unique(query_idx)

    if len(query_idx) == 0:
        print("No indices were selected, which may indicate an issue with the query function or pool.")
        continue

    learner.teach(X_pool[query_idx], y_pool[query_idx])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

    y_pred = learner.predict(X_test_np)
    accuracy = accuracy_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred, average='macro')
    acc_test_data.append(accuracy)
    f1_test_data.append(f1)
    print(f"Accuracy after query {i + 1}: {accuracy}")
    print(f"F1 Score after query {i + 1}: {f1}")

    # Update the best F1 score and handle early stopping
    if f1 > best_f1_score:
        best_f1_score = f1
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Stopping early after {i + 1} queries due to no improvement in F1 score.")
            break

    total_samples += len(query_idx)
    print(f"Total samples used for training after query {i + 1}: {total_samples}")
    power += 0.25  # Increment power for the next step
    torch.cuda.empty_cache()

    # Save checkpoint after each query
    state_filename = os.path.join(save_dir, f"al_state_query_{i+1}.pkl")
    torch.save({
        'learner_state_dict': learner.estimator.module_.state_dict(),
        'query_idx': query_idx,
        'iteration': i,
        'total_samples': total_samples,
        'best_f1_score': best_f1_score
    }, state_filename)
    print(f"Saved active learning state after query {i+1} to {state_filename}")

    '''# Plot performance
    plot_validation_performance_vs_epochs(
        performance=[{'val_loss': None, 'val_f1_score': f1}],  # Modify if you have a loss value
        no_of_samples=[total_samples],
        title="Validation Performance over Epochs",
        x_axes="Number of Queries",
        y_axes="Metric Value",
        plotting_curve="f1"
    )

    plot_validation_performance_vs_epochs(
        performance=[accuracy],
        no_of_samples=[total_samples],
        title="Validation set Accuracy using min confidence for multilabel classification",
        x_axes="Number of Queries",
        y_axes="Validation Accuracy",
        plotting_curve="accuracy"
    )

    plot_performance_vs_amount_of_data(
        performance=f1_test_data,
        no_of_samples=[total_samples],
        title="Test set performance using min confidence for multilabel classification",
        x_axes="Amount of training samples (log scaled)",
        y_axes="Test set F1 score",
        mode="testing"
    )

    plot_performance_vs_amount_of_data(
        performance=acc_test_data,
        no_of_samples=[total_samples],
        title="Validation performance using min confidence for multilabel classification",
        x_axes="Amount of training samples (log scaled)",
        y_axes="Validation set F1 score",
        mode="validation"
    )'''

print(f"Final number of samples used for training: {total_samples}")
print(f"Best F1 score across iterations: {best_f1_score}")

# Plot performance
plot_validation_performance_vs_epochs(
    performance=[{'val_loss': None, 'val_f1_score': f1}],  # Modify if you have a loss value
    no_of_samples=[total_samples],
    title="Validation Performance over Epochs",
    x_axes="Number of Queries",
    y_axes="Metric Value",
    plotting_curve="f1"
)

plot_validation_performance_vs_epochs(
    performance=[accuracy],
    no_of_samples=[total_samples],
    title="Validation set Accuracy using min confidence for multilabel classification",
    x_axes="Number of Queries",
    y_axes="Validation Accuracy",
    plotting_curve="accuracy"
)

plot_performance_vs_amount_of_data(
    performance=f1_test_data,
    no_of_samples=[total_samples],
    title="Test set performance using min confidence for multilabel classification",
    x_axes="Amount of training samples (log scaled)",
    y_axes="Test set F1 score",
    mode="testing"
)

plot_performance_vs_amount_of_data(
    performance=acc_test_data,
    no_of_samples=[total_samples],
    title="Validation performance using min confidence for multilabel classification",
    x_axes="Amount of training samples (log scaled)",
    y_axes="Validation set F1 score",
    mode="validation"
)
