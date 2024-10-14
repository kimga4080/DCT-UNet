# DCT-UNet
Dual Convolution-Transformer UNet

## Getting Started
In order to train the DCT-UNet on your dataset, you will have to generate folder of dataset like this.


## Data Split and Load
'data_spliter.py': This file splits original dataset into train, validation, and test dataset. It expects datasets in a specific structured format as follow:

    DCTUNet/Dataset/
    ├── 001
    │   ├── image.nii
    │   └── label.nii
    ├── 002
    │   ├── image.nii
    │   └── label.nii
    ├── 003
    │   ├── image.nii
    │   └── label.nii
    ├── 004
    │   ├── image.nii
    │   └── label.nii
    └── 005
        ├── image.nii
        └── label.nii
