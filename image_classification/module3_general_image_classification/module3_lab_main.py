import os 
import tarfile
import matplotlib.pyplot as plt
import torch
import numpy as np
import helper_utils
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import requests
from tqdm import tqdm
import scipy 
from PIL import Image
from torchvision import transforms


def download_dataset(): 

    data_dir = 'flower_data'
    # Define paths for key files and folders.
    image_folder_path = os.path.join(data_dir, "jpg")
    labels_file_path = os.path.join(data_dir, "imagelabels.mat")
    tgz_path = os.path.join(data_dir, "102flowers.tgz")

    if os.path.exists(image_folder_path) and os.path.exists(labels_file_path):
        # Inform the user that the dataset is already available locally.
        print(f"Dataset already exists. Loading locally from '{data_dir}'.")
        # Exit the function since no download is needed.
        return
    
        # Inform the user that the dataset is not found and the download will start.
    print("Dataset not found locally. Downloading...")
    # Create the data directory if it doesn't exist.
    # Define the URLs for the image archive and the labels file.
    image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    # Create the target directory for the dataset, if it doesn't already exist.
    os.makedirs(data_dir, exist_ok=True)

    # Announce the start of the image download process.
    print("Downloading images...")
    # Send an HTTP GET request to the image URL, enabling streaming for large files.
    response = requests.get(image_url, stream=True)
    # Get the total size of the file from the response headers for the progress bar.
    total_size = int(response.headers.get("content-length", 0)) 

    # Open a local file in binary write mode to save the downloaded archive.
    with open(tgz_path, "wb") as file:
        # Iterate over the response content in chunks with a progress bar.
        for data in tqdm(
            # Define the chunk size for iterating over the content.
            response.iter_content(chunk_size=1024),
            # Set the total for the progress bar based on the file size in kilobytes.
            total=total_size // 1024,
        ):
            # Write each chunk of data to the file.
            file.write(data)
    # Announce the start of the file extraction process.
    print("Extracting files...")
    # Open the downloaded tar.gz archive in read mode.
    with tarfile.open(tgz_path, "r:gz") as tar:
        # Extract all contents of the archive into the target directory.
        tar.extractall(data_dir)
    # Announce the start of the labels download process.
    print("Downloading labels...")
    # Send an HTTP GET request to the labels URL.
    response = requests.get(labels_url)
    # Open a local file in binary write mode to save the labels.
    with open(labels_file_path, "wb") as file:
        # Write the entire content of the response to the file.
        file.write(response.content)

    # Inform the user that the download and extraction are complete.
    print(f"Dataset downloaded and extracted to '{data_dir}'.")

# download_dataset()

# Define the path to the root directory of the dataset.
path_dataset = './flower_data'

# Display the folder structure of the dataset directory up to a depth of one.
helper_utils.print_data_folder_structure(path_dataset, max_depth=1)

class FlowerDataset(Dataset):
    """
    A custom dataset class for loading flower image data.

    This class is designed to work with PyTorch's Dataset and DataLoader
    abstractions. It handles loading images and their corresponding labels
    from a specific directory structure.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset object.

        Args:
            root_dir (str): The root directory where the dataset is stored.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Store the root directory path.
        self.root_dir = root_dir
        # Store the optional transformations.
        self.transform = transform
        # Construct the full path to the image directory.
        self.image_dir = os.path.join(self.root_dir, "jpg")
        # Load and process the labels from the corresponding file.
        self.labels = self.load_and_correct_labels()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        # The total number of samples is the number of labels.
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and its label.
        """
        # Retrieve the image for the given index.
        image = self.retrieve_image(idx)

        # Check if a transform is provided.
        if self.transform is not None:
            # Apply the transform to the image.
            image = self.transform(image)

        # Get the label corresponding to the index.
        label = self.labels[idx]

        # Return the processed image and its label.
        return image, label

    def retrieve_image(self, idx):
        """
        Loads a single image from disk based on its index.

        Args:
            idx (int): The index of the image to load.

        Returns:
            PIL.Image.Image: The loaded image, converted to RGB.
        """
        # Construct the image filename based on the index (e.g., 'image_00001.jpg').
        img_name = f"image_{idx + 1:05d}.jpg"
        # Construct the full path to the image file.
        img_path = os.path.join(self.image_dir, img_name)
        # Open the image file.
        with Image.open(img_path) as img:
            # Convert the image to the RGB color space and return it.
            image = img.convert("RGB")
        return image

    def load_and_correct_labels(self):
        """
        Loads labels from a .mat file and adjusts them to be zero-indexed.

        Returns:
            numpy.ndarray: An array of zero-indexed integer labels.
        """
        # Load the MATLAB file containing the labels.
        self.labels_mat = scipy.io.loadmat(
            os.path.join(self.root_dir, "imagelabels.mat")
        )
        # Extract the labels array and correct for zero-based indexing. 102 种花卉类别，标签从 1 到 102，需要减 1 变成 0 到 101
        # total array length is 8189, with values from 1 to 102 (flower categories)
        labels = self.labels_mat["labels"][0] - 1
        # Return the processed labels.
        return labels

    def get_label_description(self, label):
        """
        Retrieves the text description for a given label index.

        Args:
            label (int): The integer label.

        Returns:
            str: The corresponding text description of the label.
        """
        # Construct the path to the file containing label descriptions.
        path_labels_description = os.path.join(self.root_dir, "labels_description.txt")
        # Open the label description file for reading.
        with open(path_labels_description, "r") as f:
            # Read all lines from the file.
            lines = f.readlines()
        # Get the description for the specified label and remove leading/trailing whitespace.
        description = lines[label].strip()
        # Return the clean description.
        return description

dataset = FlowerDataset(root_dir=path_dataset)
correct_labels = dataset.load_and_correct_labels()
# Print the total number of samples in the dataset.
print(f'Number of samples in the dataset: {len(dataset)}\n')

# Define an index for a sample to retrieve.
sel_idx = 10

# Retrieve the image and label for the selected index.
img, label = dataset[sel_idx]

# Create a string detailing the image's dimensions.
img_size_info = f"Image size: {img.size}"

# Print the image size information along with its corresponding label.
print(f'{img_size_info}, Label: {label}\n')

helper_utils.plot_img(img, label=label, info=img_size_info)

print('Hellow World!')