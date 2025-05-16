import os
import pydicom
import nibabel as nib
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MedicalDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = None

        if file_path.endswith('.dcm'):
            image = load_dicom_slice(file_path)
        elif file_path.endswith('.nii.gz'):
            image = load_nifti_volume(file_path)
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load with PIL, convert to grayscale, then to numpy array
            image = Image.open(file_path).convert('L')
            image = np.array(image).astype(np.float32) / 255.0  # Normalize to 0-1
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label



def load_dicom_slice(path):
    """Load single DICOM slice"""
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Prevent division by zero
    return img

def load_nifti_volume(path):
    """Load 3D NIfTI volume and extract middle slice"""
    img = nib.load(path)
    data = img.get_fdata()
    data = np.rot90(data, k=1, axes=(0,1))  # Correct orientation
    middle_slice = data[..., data.shape[-1]//2]  # Take middle slice
    return middle_slice

def create_datasets(data_path, test_size=0.2):
    # Collect all file paths
    file_paths = []
    labels = []
    
    for label, class_name in enumerate(['normal', 'abnormal']):
        class_dir = os.path.join(data_path, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            file_paths.append(file_path)
            labels.append(label)
    
    # Split dataset
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels
    )
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return (train_paths, train_labels), (test_paths, test_labels), (train_transform, test_transform)
