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
from modAL.multilabel import min_confidence,avg_confidence
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
import csv

# Define image directory and load dataframes
image_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/1_all_winding_images/")
len(os.listdir(image_dir))

# df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa045h/labelling/1_all_winding_images/")

df_dir = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/2_labels/Updated_Labels/")
#train_df = pd.read_csv(df_dir + "/Splits_v2024-03-18/train_v2024-03-18_10%.csv")
train_df = pd.read_csv(df_dir + "/train_v2024-03-18.csv")
val_df = pd.read_csv(df_dir + "/validation_v2024-03-18.csv")
test_df = pd.read_csv(df_dir + "/test_v2024-03-18.csv")

# Path to the CSV file that contains the exact images to use for each iteration
SELECTED_IMAGES_CSV_PATH = os.path.abspath(r"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/multilabel/imagesfrommulticlass/")
selected_images_df = pd.read_csv(SELECTED_IMAGES_CSV_PATH + "/sample_selection_log.csv")

def batch(number, base):
    return base * round(number / base)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images(image_dir, df):
    images, labels = [], []
    for i in tqdm(range(df.shape[0])):
        image_path = os.path.join(image_dir, df.loc[i, "image"])
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        images.append(img)
        # Collect the multi-label data
        labels.append(df.drop(columns=['image']).iloc[i].values.astype('float32'))
    return torch.stack(images), torch.tensor(labels)

# Load datasets
X_train_initial, y_train_initial = load_images(image_dir, train_df)
X_val, y_val = load_images(image_dir, val_df)
X_test, y_test = load_images(image_dir, test_df)

def create_data_loader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

BATCH_SIZE = 4
initial_dataset, pool_dataset = random_split(TensorDataset(X_train_initial, y_train_initial), [100, len(X_train_initial) - 100])

train_loader = DataLoader(initial_dataset, batch_size=BATCH_SIZE, shuffle=True)
pool_loader = DataLoader(pool_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = create_data_loader(X_val, y_val, BATCH_SIZE)
test_loader = create_data_loader(X_test, y_test, BATCH_SIZE)
# CSV Logger
class CSVLogger(Callback):
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

run_id = time.strftime("%Y%m%d-%H%M%S")
SAVE_DIR = f"/home/woody/iwfa/iwfa044h/CleanLab_Test/ActiveLearningApproaches/results/multilabel/imagesfrommulticlass/{run_id}"
os.makedirs(SAVE_DIR, exist_ok=True)

# Fieldnames
FIELDNAMES = ['epoch', 'train_f1', 'train_loss', 'valid_acc', 'valid_f1', 'valid_loss', 'dur']
csv_logger = CSVLogger(os.path.join(SAVE_DIR, "training_history.csv"), FIELDNAMES)


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

# Load DINO model
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

for param in list(dino_model.parameters())[:65]:
    param.requires_grad = False

# Define model with 3 output classes
model = CustomDINONormModel(dino_model, num_classes=3).to(device)

# Ensure that the loaded labels have 3 columns
def load_images(image_dir, df):
    images, labels = [], []
    for i in tqdm(range(df.shape[0])):
        image_path = os.path.join(image_dir, df.loc[i, "image"])
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        images.append(img)
        # Ensure the labels have 3 columns for multi-label classification
        labels.append(df[['multi-label_double_winding', 'multi-label_gap', 'multi-label_crossing']].iloc[i].values.astype('float32'))
    return torch.stack(images), torch.tensor(labels)



# Reload the datasets to ensure they have the correct number of labels (3)
X_train_initial, y_train_initial = load_images(image_dir, train_df)
X_val, y_val = load_images(image_dir, val_df)
X_test, y_test = load_images(image_dir, test_df)
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
es = EarlyStopping(monitor='valid_loss', patience=13, lower_is_better=True)
model_checkpoint = Checkpoint(dirname=os.path.join(SAVE_DIR, 'model_checkpoints'), monitor='valid_loss_best')


# F1 Scorer
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
train_f1 = EpochScoring(f1_scorer, on_train=True, name='train_f1', lower_is_better=False)
valid_f1 = EpochScoring(f1_scorer, on_train=False, name='valid_f1', lower_is_better=False)

# Early stopping
es = EarlyStopping(monitor='valid_loss', patience=8, lower_is_better=True)
model_checkpoint = Checkpoint(dirname=os.path.join(SAVE_DIR, 'model_checkpoints'), monitor='valid_loss_best')

# Define the active learner
learner = ActiveLearner(
    estimator=NeuralNetClassifier(
        module=model,
        criterion=nn.BCEWithLogitsLoss,  # Use BCEWithLogitsLoss for multi-label classification
        optimizer=optim.SGD,
        optimizer__momentum=0.14729309193472406,
        lr=0.0001375803586556554,
        max_epochs=100,
        train_split=predefined_split(Dataset(X_val, y_val)),  # Validation set split
        device=device,
        callbacks=[train_f1, valid_f1, csv_logger, es]
    ),
    query_strategy=avg_confidence,
    X_training=X_train_initial.numpy(),
    y_training=y_train_initial.numpy()
)

# Active Learning Loop with Custom Sample Selection
n_queries = 13
best_f1_score = 0
wait = 0
patience = 13
acc_test_data = []
f1_test_data = []


'''def get_images_and_labels_for_iteration(selected_images_df, iteration, image_dir, label_df):
    # Get the selected images for this iteration
    images_for_iteration = selected_images_df[selected_images_df["Query_Iteration"] == iteration]["Selected_Image"].values
    loaded_images = []
    labels = []
    
    for img_name in images_for_iteration:
        image_path = os.path.join(image_dir, img_name)
        img = Image.open(image_path).convert("RGB")
        img = transform(img)
        loaded_images.append(img)

        # Fetch the corresponding labels from the label_df based on the image name
        label = label_df[label_df["image"] == img_name][['multi-label_double_winding', 'multi-label_gap', 'multi-label_crossing']].values
        labels.append(label)
    
    return torch.stack(loaded_images), torch.tensor(labels).squeeze()'''

def get_images_and_labels_for_iteration(selected_images_df, iteration, image_dir, label_df):
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
    
    # Convert lists to torch tensors and return
    return torch.stack(loaded_images), torch.tensor(labels).squeeze()


# Active Learning Loop with Corrected Label Handling
for i in range(n_queries):
    print(f"\nQuery {i + 1}: Using the exact images and labels from the CSV for this iteration.")

    # Load the exact images and their corresponding labels for this query iteration
    X_query, y_query = get_images_and_labels_for_iteration(selected_images_df, iteration=i + 1, image_dir=image_dir, label_df=train_df)

    # Convert to numpy
    X_query_np = X_query.numpy()
    y_query_np = y_query.numpy()

    # Update the cumulative datasets
    if i == 0:
        X_cumulative = X_query_np
        y_cumulative = y_query_np
    else:
        X_cumulative = np.vstack((X_cumulative, X_query_np))
        y_cumulative = np.vstack((y_cumulative, y_query_np))  # Use vstack to match dimensions

    # Teach the learner with the new data
    learner.teach(X=X_cumulative, y=y_cumulative)

    # Evaluate on test set
    y_pred = learner.predict(X_test.numpy())
    y_pred = torch.sigmoid(torch.tensor(y_pred)) > 0.5  # Use thresholding for multi-label prediction
    accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
    f1 = f1_score(y_test.numpy(), y_pred.numpy(), average='macro')
    acc_test_data.append(accuracy)
    f1_test_data.append(f1)

    print(f"Accuracy after query {i + 1}: {accuracy}")
    print(f"F1 Score after query {i + 1}: {f1}")

    if f1 > best_f1_score:
        best_f1_score = f1
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Stopping early due to no improvement in F1 score.")
            break

print(f"Best F1 score across iterations: {best_f1_score}")