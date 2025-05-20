import os
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zipfile
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb # Assuming wandb is installed and configured

# --- Constants ---
# Define constants directly or load from a config file if preferred
DATA_PATH = "path/to/data" # Update this path
MODEL_SAVE_PATH = "best_model.pt" # Path to save the best model
LOG_DIR = "logs" # Directory for logs
ZIP_PATH = 'path/to/zipfile.zip' # Update this path
CSV_FILENAME = 'filename.csv' # Update this filename
UNLABELED_PATH = 'path/to/unlabeled.csv' # Update this path

# Model Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 256
DROPOUT_PROBABILITY = 0.5

# Active Learning Hyperparameters
ACQUISITION_SIZE = 100
N_MC_DROPOUT_ITERATIONS = 10
N_ACTIVE_LEARNING_ITERATIONS = 10

# WandB Configuration
WANDB_PROJECT = "DeepDockingProject" # Update with your WandB project name

# --- Logging Setup ---
def setup_logging(log_dir=LOG_DIR):
    """Set up logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'logfile.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


# --- Utility Functions (includes visualization functions) ---
def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def read_zip_to_dataframe(zip_path, csv_filename):
    """Read a CSV file from a ZIP archive into a DataFrame."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)
                logging.info(f"Loaded dataframe from {csv_filename} in {zip_path}")
                return df
    except Exception as e:
        logging.error(f"Error reading {csv_filename} from {zip_path}: {e}")
        return None

def get_morgan(smile):
    """Get Morgan fingerprints for a given SMILE string."""
    try:
        # Using radius 2 and 1024 bits as seen in data_processor.py
        fingerprint = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=1024, useChirality=True))
        return fingerprint
    except Exception as e:
        # Log the error and return None for invalid SMILES
        logging.error(f"Error in generating Morgan fingerprint for SMILE: {smile}, Error: {e}")
        return None

def process_batch(smiles_batch):
    """Process a batch of SMILES strings to generate Morgan fingerprints."""
    # Filter out None results from get_morgan
    return [fp for fp in [get_morgan(smile) for smile in smiles_batch] if fp is not None]

def str_2_int_list(str_list):
    """Convert a string representation of a list to an actual list of integers."""
    int_list = [int(x.strip("' ")) for x in str_list.strip("[]").split(",")]
    return int_list

def log_wandb_metrics(iteration, training_losses, validation_losses, mean_mse_scores=None):
    """Log performance metrics to WandB."""
    metrics = {
        'iteration': iteration,
        'training_loss': training_losses[-1],
        'validation_loss': validation_losses[-1],
    }
    if mean_mse_scores is not None and len(mean_mse_scores) > 0:
         metrics['mean_mse'] = mean_mse_scores[-1]
    if wandb.run is not None: # Check if wandb is initialized
        wandb.log(metrics)
    else:
        logging.warning("WandB is not initialized. Skipping logging metrics.")


def plot_losses(training_losses, validation_losses):
    """Plot the training and validation losses over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter(y_test, y_predict):
    """Plot a scatter plot of true values vs. predicted values."""
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_predict, alpha=0.5)
    plt.title('True Values vs. Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.show()

# --- Dataset and DataLoader (from data_loader.py and utils.py) ---
class DockingDataset(Dataset):
    """
    Custom Dataset for docking data.
    """
    def __init__(self, X, y):
        """
        Initialize the dataset with features and targets.

        :param X: Features (numpy array or torch tensor).
        :param y: Targets (numpy array or torch tensor).
        """
        # Ensure data is torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get an item by index.

        :param idx: Index of the item.
        :return: Tuple of feature and target.
        """
        return self.X[idx], self.y[idx]

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """
    Create PyTorch DataLoaders for training and validation datasets.

    :param X_train: Training features.
    :param y_train: Training targets.
    :param X_val: Validation features.
    :param y_val: Validation targets.
    :param batch_size: Batch size for DataLoader.
    :return: Training and validation DataLoaders.
    """
    train_dataset = DockingDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = DockingDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader

# --- Data Processing (from data_processor.py) ---
class DataProcessor:
    def __init__(self, batch_size=BATCH_SIZE):
        """
        Initialize the DataProcessor with a specified batch size.

        :param batch_size: Number of samples per batch for processing.
        """
        self.batch_size = batch_size
        logging.info(f"DataProcessor initialized with batch size: {batch_size}")

    def compute_fingerprints_in_batches(self, df, label):
        """
        Compute Morgan fingerprints for the molecules in the dataframe in batches.

        :param df: DataFrame containing SMILE strings.
        :param label: Label to indicate whether data is labeled or unlabeled.
        :return: Numpy array of Morgan fingerprints.
        """
        num_batches = int(np.ceil(len(df) / self.batch_size))
        smiles_batches = [df.SMILES[i*self.batch_size:(i+1)*self.batch_size] for i in range(num_batches)]

        logging.info(f"Processing {label} data in {num_batches} batches using {cpu_count()} CPU cores...")

        results = []
        try:
            # Use process_batch function from utilities
            with Pool(cpu_count()) as pool:
                for i, batch_result in enumerate(pool.imap(process_batch, smiles_batches)):
                    results.extend(batch_result)
                    logging.info(f"Processed batch {i+1}/{num_batches} for {label}")
        except Exception as e:
            logging.error(f"Error during multiprocessing: {e}")
            return np.array([])

        # Ensure all results are numpy arrays before stacking
        fingerprints = np.array(results)
        return fingerprints

    def load_and_process_data(self, zip_path=ZIP_PATH, csv_filename=CSV_FILENAME, unlabeled_path=UNLABELED_PATH):
        """
        Load and process labeled and unlabeled data, generating Morgan fingerprints for each.

        :param zip_path: Path to the ZIP file containing labeled data.
        :param csv_filename: Name of the CSV file inside the ZIP archive.
        :param unlabeled_path: Path to the CSV file containing unlabeled data.
        :return: Tuple containing Morgan fingerprints and docking scores for labeled and unlabeled data.
                 Returns None, None, None, None if any step fails.
        """
        try:
            labeled_df = read_zip_to_dataframe(zip_path, csv_filename)
            if labeled_df is None:
                logging.error("Failed to load labeled data.")
                return None, None, None, None

            # Drop rows with missing docking score or SMILES in labeled data
            labeled_df.dropna(subset=['r_i_docking_score', 'SMILES'], inplace=True)
            logging.info(f"Labeled data shape after dropping NaNs: {labeled_df.shape}")

            unlabeled_df = pd.read_csv(unlabeled_path)
            # Drop rows with missing SMILES in unlabeled data
            unlabeled_df.dropna(subset=['SMILES'], inplace=True)
            logging.info(f"Unlabeled data shape after dropping NaNs: {unlabeled_df.shape}")

            # Merge unlabeled data with labeled data to get known docking scores for evaluation purposes
            # Use a left merge to keep all unlabeled data, and add a column for known scores
            merged_df = unlabeled_df.merge(labeled_df[['ZINCID', 'r_i_docking_score']], on='ZINCID', how='left', suffixes=('_unlabeled', '_labeled'))
            logging.info(f"Merged data shape: {merged_df.shape}")

            # Extract known docking scores for the unlabeled pool where available (for evaluation)
            # The column name will be 'r_i_docking_score_labeled' due to suffixes
            if 'r_i_docking_score_labeled' in merged_df.columns:
                y_unlabeled_known = merged_df.r_i_docking_score_labeled.values
                logging.info(f"Extracted known labels for {np.sum(~np.isnan(y_unlabeled_known))} samples in the unlabeled pool.")
            else:
                logging.warning("Could not find 'r_i_docking_score_labeled' column after merging. Cannot evaluate unlabeled pool.")
                y_unlabeled_known = np.full(len(merged_df), np.nan) # Create an array of NaNs if column not found

            logging.info("Processing labeled data...")
            X_labeled = self.compute_fingerprints_in_batches(labeled_df, "labeled")
            y_labeled = labeled_df.r_i_docking_score.values

            logging.info("Processing unlabeled data pool...")
            X_unlabeled = self.compute_fingerprints_in_batches(merged_df, "unlabeled") # Use merged_df for processing

            # Ensure that the number of fingerprints matches the number of rows in the dataframe
            if len(X_labeled) != len(labeled_df):
                 logging.warning(f"Mismatch in labeled data: {len(X_labeled)} fingerprints vs {len(labeled_df)} rows. This might indicate issues with SMILES processing.")
                 # Filter out corresponding labels if fingerprints were not generated
                 valid_labeled_smiles_indices = [i for i, smile in enumerate(labeled_df.SMILES) if get_morgan(smile) is not None]
                 y_labeled = y_labeled[valid_labeled_smiles_indices]
                 labeled_df = labeled_df.iloc[valid_labeled_smiles_indices] # Keep only rows for which fingerprints were generated
                 logging.warning(f"Adjusted labeled data shape: {X_labeled.shape}, {y_labeled.shape}")


            if len(X_unlabeled) != len(merged_df):
                 logging.warning(f"Mismatch in unlabeled data: {len(X_unlabeled)} fingerprints vs {len(merged_df)} rows. This might indicate issues with SMILES processing.")
                 # Filter out corresponding known labels if fingerprints were not generated
                 valid_unlabeled_smiles_indices = [i for i, smile in enumerate(merged_df.SMILES) if get_morgan(smile) is not None]
                 y_unlabeled_known = y_unlabeled_known[valid_unlabeled_smiles_indices]
                 merged_df = merged_df.iloc[valid_unlabeled_smiles_indices] # Keep only rows for which fingerprints were generated
                 logging.warning(f"Adjusted unlabeled data shape: {X_unlabeled.shape}, {y_unlabeled_known.shape}")


            logging.info("Data loading and processing complete.")
            return X_labeled, y_labeled, X_unlabeled, y_unlabeled_known

        except FileNotFoundError as e:
            logging.error(f"File not found: {e}. Please check your ZIP_PATH, CSV_FILENAME, and UNLABELED_PATH.")
            return None, None, None, None
        except Exception as e:
            logging.error(f"An unexpected error occurred during data loading and processing: {e}")
            return None, None, None, None


    def sample_and_split_data(self, X_labeled, y_labeled, sample_size=100000):
        """
        Sample and split the labeled data into training and validation sets.

        :param X_labeled: Labeled data features.
        :param y_labeled: Labeled data targets.
        :param sample_size: Number of samples to draw from the labeled data.
        :return: Training and validation datasets (X_train, X_val, y_train, y_val).
                 Returns None, None, None, None if input is invalid or sample size is too large.
        """
        if X_labeled is None or y_labeled is None or len(X_labeled) == 0 or len(y_labeled) == 0:
            logging.warning("No labeled data available for sampling and splitting.")
            return None, None, None, None

        np.random.seed(42)
        sample_size = min(sample_size, len(X_labeled))

        if sample_size == 0:
             logging.warning("Sample size is zero. Cannot sample and split data.")
             return None, None, None, None

        sample_indices = np.random.choice(len(X_labeled), sample_size, replace=False)
        X_sampled = X_labeled[sample_indices]
        y_sampled = y_labeled[sample_indices]

        logging.info(f"Sampled {sample_size} data points for initial training and validation.")

        # Ensure there are enough samples for both training and validation splits
        if sample_size < 2:
             logging.warning(f"Sample size ({sample_size}) is too small for splitting.")
             return None, None, None, None
        if sample_size * 0.2 < 1: # Ensure at least one sample in validation set
             logging.warning(f"Sample size ({sample_size}) is too small to create a validation set.")
             # In this case, just return the sampled data as training data and empty validation sets
             return X_sampled, np.array([]), y_sampled, np.array([])


        X_train, X_val, y_train, y_val = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"y_train shape: {y_train.shape}")
        logging.info(f"X_val shape: {X_val.shape}")
        logging.info(f"y_val shape: {y_val.shape}")

        return X_train, X_val, y_train, y_val

# --- Model Definition (from train_model.py) ---
class DockingModel(nn.Module):
    """
    Neural network model for docking.
    """
    def __init__(self, input_size, p=DROPOUT_PROBABILITY):
        """
        Initialize the DockingModel.

        :param input_size: Number of input features.
        :param p: Dropout probability for Monte Carlo Dropout.
        """
        super(DockingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 2000)
        self.fc2 = nn.Linear(2000, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input tensor.
        :return: Output tensor.
        """
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def mc_dropout_forward(self, x, n_iter=N_MC_DROPOUT_ITERATIONS):
        """
        Perform Monte Carlo Dropout forward passes.

        :param x: Input tensor.
        :param n_iter: Number of Monte Carlo iterations.
        :return: Mean and standard deviation of the predictions.
        """
        self.train()  # Set the model to training mode to apply dropout
        predictions = torch.stack([self.forward(x) for _ in range(n_iter)], dim=0)
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        self.eval()  # Set the model back to evaluation mode
        return mean_prediction, std_prediction

# --- Model Training (from train_model.py) ---
def train_epoch(model, train_dataloader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.

    :param model: The model to train.
    :param train_dataloader: DataLoader for training data.
    :param criterion: Loss function.
    :param optimizer: Optimizer.
    :param device: Device to train on ('cpu' or 'cuda').
    :param epoch: Current epoch number.
    :return: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    # Use tqdm for a progress bar
    for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Ensure labels have the same shape as outputs for loss calculation
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_training_loss = running_loss / len(train_dataloader)
    return avg_training_loss

def evaluate_model(model, val_dataloader, criterion, device):
    """
    Evaluate the model on the validation set.

    :param model: The model to evaluate.
    :param val_dataloader: DataLoader for validation data.
    :param criterion: Loss function.
    :param device: Device to evaluate on ('cpu' or 'cuda').
    :return: Average validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Ensure labels have the same shape as outputs for loss calculation
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()
    avg_validation_loss = val_loss / len(val_dataloader)
    return avg_validation_loss

def train_model(model, train_dataloader, val_dataloader, num_epochs=75, learning_rate=LEARNING_RATE, device='cpu', model_save_path=MODEL_SAVE_PATH):
    """
    Train the docking model.

    :param model: The docking model to train.
    :param train_dataloader: DataLoader for training data.
    :param val_dataloader: DataLoader for validation data.
    :param num_epochs: Number of epochs to train.
    :param learning_rate: Learning rate for the optimizer.
    :param device: Device to train on ('cpu' or 'cuda').
    :param model_save_path: Path to save the best model checkpoint.
    :return: Tuple of training losses and validation losses.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    validation_losses = []

    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # Early stopping patience

    logging.info(f"Starting training for {num_epochs} epochs on {device}")

    for epoch in range(num_epochs):
        # Train epoch
        avg_training_loss = train_epoch(model, train_dataloader, criterion, optimizer, device, epoch)
        training_losses.append(avg_training_loss)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_training_loss:.4f}")

        # Evaluate model
        avg_validation_loss = evaluate_model(model, val_dataloader, criterion, device)
        validation_losses.append(avg_validation_loss)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_validation_loss:.4f}")

        # Early stopping logic
        if avg_validation_loss < best_val_loss:
            best_val_loss = avg_validation_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)  # Save the best model
            logging.info(f"Saved best model to {model_save_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss did not improve. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logging.info('Early stopping triggered!')
            model.load_state_dict(torch.load(model_save_path))  # Load the best model
            logging.info('Loaded best model state dict.')
            break

        # Log metrics using WandB if initialized
        if wandb.run is not None:
             log_wandb_metrics(epoch, training_losses, validation_losses)


    logging.info("Training completed.")
    return training_losses, validation_losses

# --- Model Prediction (from predict_model.py) ---
def predict_model(model, dataloader, device, acquisition_function='greedy', n_mc_dropout=N_MC_DROPOUT_ITERATIONS):
    """
    Predict the labels for the given dataloader using the provided model.

    :param model: The trained model to use for predictions.
    :param dataloader: DataLoader containing the data for predictions.
    :param device: Device to perform the computations on ('cpu' or 'cuda').
    :param acquisition_function: Acquisition function to use ('greedy', 'mc_dropout').
    :param n_mc_dropout: Number of Monte Carlo Dropout iterations (only used if acquisition_function is 'mc_dropout').
    :return: Array of predictions and optionally uncertainties (if acquisition_function is 'mc_dropout').
    """
    model.eval()
    predictions = []
    uncertainties = []

    logging.info(f"Making predictions using {acquisition_function} acquisition function...")

    with torch.no_grad():
        # Use tqdm for a progress bar
        for inputs, _ in tqdm(dataloader, desc="Predicting"): # Assuming dataloader yields inputs and dummy labels
            inputs = inputs.to(device)

            if acquisition_function == 'mc_dropout':
                # Need to ensure the model class has the mc_dropout_forward method
                if hasattr(model, 'mc_dropout_forward'):
                    mean, std = model.mc_dropout_forward(inputs, n_iter=n_mc_dropout)
                    predictions.extend(mean.cpu().numpy().flatten()) # Flatten to 1D array
                    uncertainties.extend(std.cpu().numpy().flatten()) # Flatten to 1D array
                else:
                    logging.error("Model does not have mc_dropout_forward method for MC Dropout.")
                    # Fallback to standard prediction if MC Dropout method is missing
                    outputs = model(inputs)
                    predictions.extend(outputs.cpu().numpy().flatten())
                    uncertainties = None # No uncertainties if MC Dropout is not performed
            else:  # Default to standard prediction (greedy, random, EDL might use this)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy().flatten())
                uncertainties = None # No uncertainties for standard prediction

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties) if uncertainties is not None else None

    logging.info("Prediction complete.")

    if acquisition_function == 'mc_dropout' and uncertainties is not None:
        return predictions, uncertainties
    else:
        return predictions

# --- Active Learning (from active_learning.py) ---
class DockingModelActiveLearning(DockingModel):
    """
    Active Learning class for the Docking Model.
    Inherits from DockingModel and adds active learning specific methods.
    """

    def __init__(self, input_size, device, acquisition_function='greedy', n_mc_dropout=N_MC_DROPOUT_ITERATIONS):
        """
        Initialize the DockingModelActiveLearning.

        :param input_size: Number of input features.
        :param device: Device to train on ('cpu' or 'cuda').
        :param acquisition_function: Acquisition function for selecting samples ('greedy', 'mc_dropout', 'random', 'EDL').
        :param n_mc_dropout: Number of Monte Carlo iterations.
        """
        # Initialize the base model
        super().__init__(input_size, p=DROPOUT_PROBABILITY)
        self.device = device
        self.acquisition_function = acquisition_function
        self.n_mc_dropout = n_mc_dropout
        self.training_losses = []
        self.validation_losses = []
        self.mean_mse_scores = [] # To store MSE on selected samples

        # Move model to device
        self.to(device)


    def inference(self, data_tensor):
        """
        Perform standard inference on a tensor of data.

        :param data_tensor: Input data tensor.
        :return: Predictions as a numpy array.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(data_tensor)
        return outputs.cpu().numpy().flatten() # Return as numpy array


    def select_top_s(self, predictions, uncertainties, S):
        """
        Select the top S samples based on the acquisition function.

        :param predictions: Predictions for the unlabeled data (numpy array).
        :param uncertainties: Uncertainty estimates (numpy array, optional for non-MC Dropout acquisition functions).
        :param S: Number of samples to select (acquisition size).
        :return: Indices of the selected samples (numpy array).
        """
        logging.info(f"Selecting top {S} samples using '{self.acquisition_function}' acquisition function.")
        if self.acquisition_function == 'greedy':
            # Select samples with the lowest predicted docking score (most promising)
            selected_indices = np.argsort(predictions)[:S]
        elif self.acquisition_function == 'random':
            # Select random samples
            selected_indices = np.random.choice(len(predictions), S, replace=False)
        elif self.acquisition_function == 'mc_dropout' and uncertainties is not None:
            # Select samples with the highest uncertainty
            selected_indices = np.argsort(uncertainties)[-S:]
        elif self.acquisition_function == 'EDL' and uncertainties is not None:
             # Assuming EDL also uses uncertainty for selection (similar to MC Dropout)
             selected_indices = np.argsort(uncertainties)[-S:]
        else:
            raise ValueError(f"Unknown or unsupported acquisition function: {self.acquisition_function}. "
                             "If using 'mc_dropout' or 'EDL', ensure uncertainties are provided.")

        logging.info(f"Selected {len(selected_indices)} samples.")
        return selected_indices


    def compute_mean_mse(self, predicted_scores, true_scores):
        """
        Compute and store the Mean Squared Error between predicted and true scores.

        :param predicted_scores: Predicted docking scores (numpy array).
        :param true_scores: True docking scores (numpy array).
        """
        # Only compute MSE for samples where true scores are available (not NaN)
        valid_indices = ~np.isnan(true_scores)
        if np.sum(valid_indices) > 0:
            mse = np.mean((predicted_scores[valid_indices] - true_scores[valid_indices])**2)
            self.mean_mse_scores.append(mse)
            logging.info(f"Mean MSE on selected samples with known labels: {mse:.4f}")
        else:
            logging.warning("No known true scores available for selected samples to compute MSE.")
            self.mean_mse_scores.append(np.nan) # Append NaN if no valid scores


    def update_training_set(self, X_train, y_train, X_unlabeled, y_unlabeled_known, selected_indices):
        """
        Add the selected samples and their true labels to the training set
        and remove them from the unlabeled pool.

        :param X_train: Current training data features (numpy array).
        :param y_train: Current training data labels (numpy array).
        :param X_unlabeled: Current unlabeled data features (numpy array).
        :param y_unlabeled_known: Known labels for the unlabeled pool (numpy array).
        :param selected_indices: Indices of samples selected from the unlabeled pool.
        :return: Updated X_train, y_train, X_unlabeled, y_unlabeled_known.
        """
        # Get the selected samples and their known labels
        X_selected = X_unlabeled[selected_indices]
        y_selected = y_unlabeled_known[selected_indices]

        # Filter out samples from the selected batch that have NaN labels
        valid_selected_indices = ~np.isnan(y_selected)
        X_selected_valid = X_selected[valid_selected_indices]
        y_selected_valid = y_selected[valid_selected_indices]

        logging.info(f"Adding {len(X_selected_valid)} selected samples (with known labels) to the training set.")

        # Add the valid selected samples to the training set
        X_train_new = np.vstack((X_train, X_selected_valid))
        y_train_new = np.concatenate((y_train, y_selected_valid))

        # Remove the selected samples (regardless of whether their label was known) from the unlabeled pool
        unselected_mask = np.ones(len(X_unlabeled), dtype=bool)
        unselected_mask[selected_indices] = False
        X_unlabeled_new = X_unlabeled[unselected_mask]
        y_unlabeled_known_new = y_unlabeled_known[unselected_mask]

        logging.info(f"Training set size after update: {len(X_train_new)}")
        logging.info(f"Unlabeled pool size after update: {len(X_unlabeled_new)}")

        return X_train_new, y_train_new, X_unlabeled_new, y_unlabeled_known_new

    def retrain_model(self, X_train, y_train, num_retrain_epochs):
        """
        Retrain the model with the updated training set.

        :param X_train: Updated training data features (numpy array).
        :param y_train: Updated training data labels (numpy array).
        :param num_retrain_epochs: Number of epochs to retrain for.
        """
        logging.info(f"Retraining model for {num_retrain_epochs} epochs with updated training set.")
        # Create DataLoader for the updated training set
        # Note: No validation set used during retraining in this active learning loop
        train_dataset = DockingDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Define optimizer and criterion for retraining
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        # Retrain the model
        for epoch in range(num_retrain_epochs):
            avg_training_loss = train_epoch(self, train_dataloader, criterion, optimizer, self.device, epoch)
            logging.info(f"Retraining Epoch {epoch+1}/{num_retrain_epochs}, Training Loss: {avg_training_loss:.4f}")

        logging.info("Retraining complete.")


    def evaluate_model_al(self, val_dataloader):
        """
        Evaluate the model on the validation set during active learning.
        Stores validation loss in the instance variable.

        :param val_dataloader: DataLoader for validation data.
        :return: Average validation loss.
        """
        criterion = nn.MSELoss()
        avg_validation_loss = evaluate_model(self, val_dataloader, criterion, self.device)
        self.validation_losses.append(avg_validation_loss)
        logging.info(f"Active Learning Iteration Validation Loss: {avg_validation_loss:.4f}")
        return avg_validation_loss


    def random_10k_evaluation(self, X_pool, y_pool_known, predictions, wandb_config):
        """
        Perform random 10K evaluation on the unlabeled pool.
        :param X_pool: The unlabeled pool features (numpy array).
        :param y_pool_known: Known labels for the unlabeled pool (numpy array).
        :param predictions: Model predictions for the unlabeled pool (numpy array).
        :param wandb_config: WandB configuration.
        :return: Pearson correlation for random 10K samples with known labels.
        """
        # Find indices where known labels are available in the pool
        known_label_indices = np.where(~np.isnan(y_pool_known))[0]

        if len(known_label_indices) < 10000:
            logging.warning(f"Only {len(known_label_indices)} samples with known labels available in the pool. Cannot perform 10K random evaluation.")
            return np.nan # Return NaN if not enough known labels

        # Randomly sample 10K indices from those with known labels
        rand_10K_indices_in_known = np.random.choice(len(known_label_indices), size=10000, replace=False)
        rand_10K_pool_indices = known_label_indices[rand_10K_indices_in_known]

        rand_10K_labels = y_pool_known[rand_10K_pool_indices]
        rand_10K_pred = predictions[rand_10K_pool_indices]

        # Ensure predictions and labels are 1D arrays for corrcoef
        rand_10K_labels = rand_10K_labels.flatten()
        rand_10K_pred = rand_10K_pred.flatten()

        pearson = np.corrcoef(rand_10K_labels, rand_10K_pred)[0, 1]
        return pearson

    def top_1percent_evaluation(self, X_pool, y_pool_known, predictions, uncertainties):
        """
        Perform top 1% evaluation on the unlabeled pool based on predictions (lowest score).
        :param X_pool: The unlabeled pool features (numpy array).
        :param y_pool_known: Known labels for the unlabeled pool (numpy array).
        :param predictions: Model predictions for the unlabeled pool (numpy array).
        :param uncertainties: Uncertainty estimates for the unlabeled pool (numpy array, optional).
        :return: Pearson correlation for top 1% samples with known labels.
        """
        if len(predictions) == 0:
             logging.warning("No predictions available for top 1% evaluation.")
             return np.nan

        # Determine the number of top samples (1% of the pool)
        num_top_samples = max(1, int(0.01 * len(predictions)))
        logging.info(f"Evaluating top {num_top_samples} samples based on predictions.")

        # Select indices of samples with the lowest predicted scores
        top_indices_in_pool = np.argsort(predictions)[:num_top_samples]

        # Filter these top indices to keep only those with known labels
        top_indices_with_known_labels = [idx for idx in top_indices_in_pool if ~np.isnan(y_pool_known[idx])]

        if len(top_indices_with_known_labels) == 0:
            logging.warning("No known labels available among the top 1% predicted samples.")
            return np.nan # Return NaN if no known labels in the top 1%

        top_1percent_labels = y_pool_known[top_indices_with_known_labels]
        top_1percent_pred = predictions[top_indices_with_known_labels]

        # Ensure predictions and labels are 1D arrays for corrcoef
        top_1percent_labels = top_1percent_labels.flatten()
        top_1percent_pred = top_1percent_pred.flatten()

        pearson = np.corrcoef(top_1percent_labels, top_1percent_pred)[0, 1]
        return pearson


    def run_active_learning(self, X_train, y_train, X_unlabeled, y_unlabeled_known, val_dataloader, K=N_ACTIVE_LEARNING_ITERATIONS, S=ACQUISITION_SIZE, num_retrain_epochs=7, wandb_config=None):
        """
        Run the active learning process.

        :param X_train: Initial training data features (numpy array).
        :param y_train: Initial training data labels (numpy array).
        :param X_unlabeled: Unlabeled data features (numpy array).
        :param y_unlabeled_known: Known labels for the unlabeled data pool (numpy array).
        :param val_dataloader: DataLoader for validation data.
        :param K: Number of active learning iterations.
        :param S: Number of samples to select in each iteration (acquisition size).
        :param num_retrain_epochs: Number of epochs to retrain in each iteration.
        :param wandb_config: WandB configuration for logging (e.g., {'metric': 'rand10K'} or {'metric': 'top1%'}).
        """
        logging.info(f"Starting active learning process for {K} iterations with acquisition size {S}.")

        # Initial training (optional, but good practice before AL loop)
        logging.info("Performing initial training...")
        # Create DataLoader for the initial training set
        initial_train_dataset = DockingDataset(X_train, y_train)
        initial_train_dataloader = DataLoader(initial_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # Use the train_model function for initial training
        initial_training_losses, initial_validation_losses = train_model(
            self, initial_train_dataloader, val_dataloader, num_epochs=num_retrain_epochs, device=self.device
        )
        self.training_losses.extend(initial_training_losses)
        self.validation_losses.extend(initial_validation_losses)
        logging.info("Initial training complete.")


        # Convert unlabeled pool to tensor once if device is GPU to avoid repeated transfers
        X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32).to(self.device)


        for iteration in range(K):
            logging.info(f"--- Active Learning Iteration {iteration+1}/{K} ---")

            # Perform inference on the entire unlabeled dataset pool.
            logging.info("Performing inference on the unlabeled pool...")
            if self.acquisition_function == 'mc_dropout' or self.acquisition_function == 'EDL':
                # Ensure the model is in training mode for dropout during MC inference
                self.train()
                mean_predictions, uncertainties = self.mc_dropout_forward(X_unlabeled_tensor, n_iter=self.n_mc_dropout)
                mean_predictions = mean_predictions.cpu().numpy().flatten()
                uncertainties = uncertainties.cpu().numpy().flatten()
                self.eval() # Set back to evaluation mode
                logging.info("Inference with MC Dropout complete.")
            else:
                # Standard inference
                mean_predictions = self.inference(X_unlabeled_tensor)
                uncertainties = None
                logging.info("Standard inference complete.")

            # Select samples based on the acquisition function.
            selected_indices_in_pool = self.select_top_s(mean_predictions, uncertainties, S)

            # Compute and store the mean MSE between predictions and true labels for selected samples (if labels are known).
            # Need to get the true values for the selected indices from the original y_unlabeled_known
            true_values_for_selected = y_unlabeled_known[selected_indices_in_pool]
            self.compute_mean_mse(mean_predictions[selected_indices_in_pool], true_values_for_selected)

            # Add the selected samples and their true labels to the training set and remove from the unlabeled pool.
            X_train, y_train, X_unlabeled, y_unlabeled_known = self.update_training_set(
                X_train, y_train, X_unlabeled, y_unlabeled_known, selected_indices_in_pool
            )

            # Update the unlabeled tensor for the next iteration
            X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32).to(self.device)

            # Retrain the model with the updated training set.
            self.retrain_model(X_train, y_train, num_retrain_epochs)

            # Evaluate the model on the validation set after retraining.
            avg_validation_loss = self.evaluate_model_al(val_dataloader)


            # Perform active learning specific evaluations (random 10K or top 1%)
            if wandb_config and wandb_config.get('metric') == 'rand10K':
                # Need to re-run inference on the *current* unlabeled pool for this evaluation
                if self.acquisition_function == 'mc_dropout' or self.acquisition_function == 'EDL':
                     self.train()
                     eval_predictions, _ = self.mc_dropout_forward(X_unlabeled_tensor, n_iter=self.n_mc_dropout)
                     eval_predictions = eval_predictions.cpu().numpy().flatten()
                     self.eval()
                else:
                     eval_predictions = self.inference(X_unlabeled_tensor)

                pearson = self.random_10k_evaluation(X_unlabeled, y_unlabeled_known, eval_predictions, wandb_config)
                logging.info(f"Pearson correlation on random 10K samples from current pool: {pearson:.4f}")
                if wandb.run is not None:
                     wandb.log({'al_iteration': iteration, 'rand10K_pearson': pearson})

            elif wandb_config and wandb_config.get('metric') == 'top1%':
                # Need to re-run inference on the *current* unlabeled pool for this evaluation
                if self.acquisition_function == 'mc_dropout' or self.acquisition_function == 'EDL':
                     self.train()
                     eval_predictions, eval_uncertainties = self.mc_dropout_forward(X_unlabeled_tensor, n_iter=self.n_mc_dropout)
                     eval_predictions = eval_predictions.cpu().numpy().flatten()
                     eval_uncertainties = eval_uncertainties.cpu().numpy().flatten()
                     self.eval()
                else:
                     eval_predictions = self.inference(X_unlabeled_tensor)
                     eval_uncertainties = None # No uncertainties for standard inference

                pearson = self.top_1percent_evaluation(X_unlabeled, y_unlabeled_known, eval_predictions, eval_uncertainties)
                logging.info(f"Pearson correlation on top 1% samples from current pool: {pearson:.4f}")
                if wandb.run is not None:
                     wandb.log({'al_iteration': iteration, 'top1%_pearson': pearson})


            # Log performance metrics to WandB for the iteration
            if wandb.run is not None:
                 log_wandb_metrics(iteration, self.training_losses, self.validation_losses, self.mean_mse_scores)


        logging.info("Active learning process completed.")

        # Plot the training and validation loss after all iterations
        self.plot_losses()


    def plot_losses(self):
        """Plot the training and validation losses accumulated during active learning."""
        plot_losses(self.training_losses, self.validation_losses)


# --- Main Execution Flow ---
if __name__ == '__main__':
    # Setup logging
    setup_logging()
    logging.info("Script started.")

    # Initialize WandB (optional, uncomment if you want to use WandB)
    # wandb.init(project=WANDB_PROJECT, config={
    #     "learning_rate": LEARNING_RATE,
    #     "batch_size": BATCH_SIZE,
    #     "dropout_probability": DROPOUT_PROBABILITY,
    #     "acquisition_size": ACQUISITION_SIZE,
    #     "n_mc_dropout_iterations": N_MC_DROPOUT_ITERATIONS,
    #     "n_active_learning_iterations": N_ACTIVE_LEARNING_ITERATIONS,
    #     "num_retrain_epochs": 7, # Example value
    #     "acquisition_function": "greedy", # Example value
    #     "evaluation_metric": "rand10K" # Example value: 'rand10K' or 'top1%'
    # })
    # wandb_config = wandb.config if wandb.run is not None else None
    wandb_config = None # Set to None if not using WandB


    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Creation/Loading and Processing ---
    processor = DataProcessor(batch_size=BATCH_SIZE)
    X_labeled, y_labeled, X_unlabeled, y_unlabeled_known = processor.load_and_process_data(
        zip_path=ZIP_PATH, csv_filename=CSV_FILENAME, unlabeled_path=UNLABELED_PATH
    )

    if X_labeled is None or X_unlabeled is None:
        logging.error("Data loading and processing failed. Exiting.")
    else:
        logging.info("Data loaded and processed successfully.")
        logging.info(f"Labeled data shape: {X_labeled.shape}, {y_labeled.shape}")
        logging.info(f"Unlabeled data pool shape: {X_unlabeled.shape}, {y_unlabeled_known.shape} (known labels in pool)")


        # --- Make Dataset (Sample and Split Labeled Data) ---
        # Sample a subset of labeled data for initial training and validation
        X_train, X_val, y_train, y_val = processor.sample_and_split_data(X_labeled, y_labeled, sample_size=100000) # Adjust sample_size as needed

        if X_train is None or X_val is None:
            logging.error("Data sampling and splitting failed. Exiting.")
        else:
            logging.info("Data sampled and split successfully.")

            # Create DataLoaders
            train_dataloader, val_dataloader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE)
            logging.info("DataLoaders created.")

            # --- Train Model (Initial Training) ---
            input_size = X_labeled.shape[1] # Input size is the fingerprint length
            model = DockingModel(input_size=input_size, p=DROPOUT_PROBABILITY)

            logging.info("Starting initial model training...")
            # Initial training using the sampled labeled data
            training_losses, validation_losses = train_model(
                model, train_dataloader, val_dataloader, num_epochs=75, # Adjust epochs as needed
                learning_rate=LEARNING_RATE, device=device, model_save_path=MODEL_SAVE_PATH
            )
            logging.info("Initial model training completed.")

            # Plot initial training and validation losses
            plot_losses(training_losses, validation_losses)

            # --- Predict Model (Example Usage after initial training) ---
            # Example: Make predictions on the validation set
            logging.info("Making predictions on the validation set...")
            # Create a DataLoader for prediction (batch size can be larger)
            val_predict_dataset = DockingDataset(X_val, y_val) # Use y_val as dummy labels if predict_model expects 2 outputs
            val_predict_dataloader = DataLoader(val_predict_dataset, batch_size=512)

            # Load the best saved model for prediction
            best_model = DockingModel(input_size=input_size, p=DROPOUT_PROBABILITY)
            best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            best_model.to(device)

            # Make predictions
            # If using MC Dropout for prediction, set acquisition_function='mc_dropout'
            predictions_val = predict_model(best_model, val_predict_dataloader, device, acquisition_function='greedy') # Or 'mc_dropout'
            logging.info("Predictions on validation set made.")

            # Visualize predictions vs true values for the validation set
            if predictions_val is not None and len(predictions_val) == len(y_val):
                plot_scatter(y_val, predictions_val)
            else:
                logging.warning("Could not plot scatter plot: Mismatch between validation labels and predictions.")


            # --- Active Learning ---
            # Ensure there is unlabeled data and known labels in the pool for evaluation
            if X_unlabeled is not None and y_unlabeled_known is not None and len(X_unlabeled) > 0:
                logging.info("Starting active learning process...")
                # Initialize the Active Learning model with the state dict from the best trained model
                al_model = DockingModelActiveLearning(
                    input_size=input_size, device=device,
                    acquisition_function='greedy', # Choose acquisition function: 'greedy', 'mc_dropout', 'random', 'EDL'
                    n_mc_dropout=N_MC_DROPOUT_ITERATIONS
                )
                al_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
                al_model.to(device)

                # Run the active learning loop
                al_model.run_active_learning(
                    X_train, y_train, X_unlabeled, y_unlabeled_known, val_dataloader,
                    K=N_ACTIVE_LEARNING_ITERATIONS, S=ACQUISITION_SIZE, num_retrain_epochs=7, # Adjust parameters
                    wandb_config=wandb_config # Pass WandB config if initialized
                )
                logging.info("Active learning process finished.")

                # Plot losses from active learning (already handled within run_active_learning)

            else:
                logging.warning("Skipping active learning: No unlabeled data or known labels available in the pool.")

    # Finish WandB run (optional)
    # if wandb.run is not None:
    #     wandb.finish()

    logging.info("Script finished.")
