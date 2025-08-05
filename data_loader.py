import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class PolygonColourDataset(Dataset):
    def __init__(self, data_dir, data_json_path, image_size=(128, 128), universal_colors=None):
        self.data_dir = data_dir
        self.inputs_dir = os.path.join(data_dir, 'inputs')
        self.outputs_dir = os.path.join(data_dir, 'outputs')
        self.image_size = image_size
        self.transform = ToTensor()
        
        try:
            with open(data_json_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = []

        if universal_colors:
            self.colors = universal_colors
        else:
            self.colors = sorted(list(set(entry['colour'] for entry in self.data)))
        
        self.color_map = {color: i for i, color in enumerate(self.colors)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        input_filename = entry['input_polygon']
        output_filename = entry['output_image']
        color_name = entry['colour']
        input_path = os.path.join(self.inputs_dir, input_filename)
        output_path = os.path.join(self.outputs_dir, output_filename)
        input_image = Image.open(input_path).convert('L').resize(self.image_size)
        output_image = Image.open(output_path).convert('RGB').resize(self.image_size)
        input_tensor = self.transform(input_image)
        output_tensor = self.transform(output_image)
        color_one_hot = torch.zeros(len(self.colors))
        if color_name in self.color_map:
            color_one_hot[self.color_map[color_name]] = 1.0

        return input_tensor, color_one_hot, output_tensor

    def get_colors(self):
        return self.colors



# The rest of your data_loader.py code is here...

# This block will run only when the script is executed directly
if __name__ == '__main__':
    # Define paths to your dataset
    DATASET_DIR = "dataset"
    TRAINING_DATA_JSON = os.path.join(DATASET_DIR, 'training', 'data.json')

    # Create an instance of the dataset class
    print("--- Testing the PolygonColourDataset class ---")
    try:
        train_dataset = PolygonColourDataset(
            data_dir=os.path.join(DATASET_DIR, 'training'),
            data_json_path=TRAINING_DATA_JSON
        )
        print("Dataset loaded successfully.")
        
        # Access one item to trigger the __getitem__ debug print
        if len(train_dataset) > 0:
            _ = train_dataset[0]
        else:
            print("Dataset is empty, cannot retrieve an item.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    print("--- Test complete ---")