# Medical Image Classification Tool

A beginner-friendly, modular deep learning tool for classifying medical images (DICOM, NIfTI, JPEG, PNG) as normal or abnormal using PyTorch.  
This project is designed for learning and experimentation, with clear code structure and detailed comments.

## Features

- Supports DICOM (`.dcm`), NIfTI (`.nii.gz`), JPEG, and PNG images
- Custom PyTorch Dataset and CNN model
- Easy-to-edit training and prediction scripts
- Works on CPU and GPU
- Modular code for easy extension

## Folder Structure

medical_image_classification_tool/
├── data/
│   ├── train/
│   │   ├── normal/
│   │   └── abnormal/
│   └── test/
│       ├── normal/
│       └── abnormal/
├── train.py
├── predict.py
├── data_preparation.py
├── model.py
├── requirements.txt
└── README.md

## Getting Started

### 1. Clone the repository

git clone https: https://github.com/YousefAlbawab/Medical_Image_Classification_Tool.git
cd medical_image_classification_tool

### 2. Install dependencies

It’s recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html):

python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt

> Note:  
> If you encounter issues with `nibabel` and NumPy, downgrade NumPy:  
> `pip install numpy==1.26.4`



### 3. Prepare your data

- Place your images in the corresponding folders:
  
  data/train/normal/
  data/train/abnormal/
  data/test/normal/
  data/test/abnormal/
  
- Supported formats: `.dcm`, `.nii.gz`, `.jpg`, `.jpeg`, `.png`
- Tip: Download free datasets from [Kaggle](https://www.kaggle.com/), [TCIA](https://www.cancerimagingarchive.net/), or [NIH Chest X-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC).

## Usage

### 1. Train the Model

python train.py

- Trains a CNN on your dataset
- Progress and accuracy are printed to the terminal
- The trained model is saved as `model.pth`

### 2. Make Predictions

python predict.py --image path/to/your/image.dcm

or edit `predict.py` to set the image path manually.

- Outputs the predicted class: normal or abnormal

## Code Overview

- `data_preparation.py`: Data loading, preprocessing, and dataset class
- `model.py`: Defines the CNN model
- `train.py`: Training loop, validation, and saving the model
- `predict.py`: Loads a trained model and predicts the class of a new image

## Customization

- Model: Edit `model.py` to change the CNN architecture
- Transforms: Adjust augmentations and preprocessing in `data_preparation.py`
- Hyperparameters: Change batch size, epochs, learning rate in `train.py`
- Add new classes: Add new folders under `data/train/` and update class lists in the code

## Troubleshooting

- Unsupported file type:  
  Make sure your images are `.dcm`, `.nii.gz`, `.jpg`, `.jpeg`, or `.png`

- NumPy/nibabel error: 
  Downgrade NumPy: `pip install numpy==1.26.4`

- CUDA out of memory:  
  Lower the batch size in `train.py`

## References & Datasets

- [COVID-19 Chest X-ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This project is for educational and research purposes.  
Please respect the licenses of any datasets you use.

## Acknowledgments

- Inspired by the open-source medical imaging community
- Thanks to [PyTorch](https://pytorch.org/) and [Open Source Datasets](https://github.com/sfikas/medical-imaging-datasets)

Happy Learning!  
If you have questions or suggestions, open an issue or discussion.
