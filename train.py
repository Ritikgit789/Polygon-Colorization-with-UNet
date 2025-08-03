import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from data_loader import PolygonColourDataset
from model import UNet
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

# Function to convert a tensor to a displayable image
to_pil = ToPILImage()

def log_images_to_wandb(inputs, outputs, targets, epoch):
    """
    Creates a grid of input, output, and target images and logs them to wandb.
    This helps visualize model performance during validation.
    """
    # Clamp output values to be in the [0, 1] range for image display
    outputs = torch.clamp(outputs, 0, 1) 
    
    # Inputs are grayscale, so we convert them to 3-channel RGB for display
    inputs_rgb = inputs.repeat(1, 3, 1, 1)
    
    # Create an empty list to hold images for a single grid
    images_to_log = []
    
    # Log a maximum of 4 examples to avoid cluttering the wandb dashboard
    for i in range(min(4, inputs.shape[0])):
        # Stack the original, generated, and ground truth images horizontally
        img_stack = torch.cat([inputs_rgb[i], outputs[i], targets[i]], dim=2)
        images_to_log.append(img_stack)

    # Make a single grid and log it to the wandb dashboard
    grid = make_grid(images_to_log, nrow=1)
    wandb.log({"validation_images": [wandb.Image(to_pil(grid))], "epoch": epoch})


def main(config):
    """
    Main function to handle the entire training process.
    """
    # Initialize wandb for experiment tracking and logging
    # wandb.init(project="unet-polygon-colorizer", config=config)
    # 'My_First_Training' 
    wandb.init(project="unet-polygon-colorizer", config=config, name="UNET_Training")

    # Set up the device for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets using the custom data loader
    train_data_json = os.path.join(config.dataset_path, 'training', 'data.json')
    val_data_json = os.path.join(config.dataset_path, 'validation', 'data.json')
    
    # First, create the training dataset to get the universal color mapping
    train_dataset = PolygonColourDataset(os.path.join(config.dataset_path, 'training'), train_data_json)
    
    # Then create the validation dataset using the same color mapping
    val_dataset = PolygonColourDataset(os.path.join(config.dataset_path, 'validation'), val_data_json, universal_colors=train_dataset.get_colors())

    print(f"Datasets loaded. Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Create data loaders for batching and shuffling the data
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    num_colors = len(train_dataset.get_colors())

    # Initialize the UNet model, loss function, and optimizer
    model = UNet(n_channels=1, n_classes=3, num_colors=num_colors).to(device)
    criterion = nn.L1Loss()  # L1Loss is often good for image generation tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Model initialized with {num_colors} colors and loaded onto {device}.")

    # Log the model's architecture and gradients to wandb
    wandb.watch(model)
    
    best_val_loss = float('inf')

    print("\nStarting training loop...")

    # Main training loop
    for epoch in range(config.epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        
        # Iterate over the training data with a progress bar
        with tqdm(train_loader, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch {epoch+1}/{config.epochs}")
            for inputs, color_one_hot, targets in t_epoch:
                # Move tensors to the selected device (GPU/CPU)
                inputs, color_one_hot, targets = inputs.to(device), color_one_hot.to(device), targets.to(device)

                # Zero the gradients from the previous step
                optimizer.zero_grad()
                
                # Forward pass: get model predictions
                outputs = model(inputs, color_one_hot)
                
                # Calculate the loss
                loss = criterion(outputs, targets)
                
                # Backward pass: compute gradients
                loss.backward()
                
                # Update model parameters
                optimizer.step()
                
                # Accumulate loss for the epoch
                train_loss += loss.item() * inputs.size(0)
                t_epoch.set_postfix(loss=loss.item())

        train_loss /= len(train_dataset)
        wandb.log({"train_loss": train_loss, "epoch": epoch})

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, color_one_hot, targets in val_loader:
                inputs, color_one_hot, targets = inputs.to(device), color_one_hot.to(device), targets.to(device)
                
                outputs = model(inputs, color_one_hot)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
            
            # Log some qualitative examples to wandb
            log_images_to_wandb(inputs, outputs, targets, epoch)
        
        val_loss /= len(val_dataset)
        wandb.log({"val_loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch+1} finished. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
    
    wandb.finish()

    print("\nTraining complete. Model saved as best_model.pth.")


if __name__ == '__main__':
    # Parse command-line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="UNet Polygon Colorizer Training")
    parser.add_argument('--dataset_path', type=str, default='dataset', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    main(args)