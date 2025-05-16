import torch
from torchvision import transforms
from .model import MedicalCNN
from .data_preparation import load_dicom_slice, load_nifti_volume

class MedicalClassifier:
    def __init__(self, model_path='../model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MedicalCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def predict(self, image_path):
        # Load image
        if image_path.endswith('.dcm'):
            image = load_dicom_slice(image_path)
        elif image_path.endswith('.nii.gz'):
            image = load_nifti_volume(image_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Preprocess
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image)
            _, prediction = torch.max(output, 1)
        
        return 'abnormal' if prediction.item() == 1 else 'normal'

# Example usage
if __name__ == "__main__":
    classifier = MedicalClassifier()
    print(classifier.predict('../data/test/abnormal/example.dcm'))
