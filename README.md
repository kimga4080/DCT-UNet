# DCT-UNet
Dual Convolution-Transformer UNet

## Getting Started
The DCT-UNet was developed and tested in a virtual environment using Anaconda. The `requirements.txt` contains the instailled package lists and their respective versions within the conda enviroment.

## Data Split and Load
`data_spliter.py` splits original dataset into train, validation, and test dataset. To use this script, your dataset folder should be organized as follows:
```
./DCTUNet/Dataset/
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
```
`data_loader.py` load the train, calidation, and test dataset. It includes pre-processing of image such as resampling, intensity scaling, and randomly cropping. Croppoing is performed for train and validation dataset.

## Model Training and Inference
`model.py` / 'block_covolution.py` / `block_transformer.py` contains functions to create the DCT-UNet.

`train.py` is used to train the DCT-UNet. It saves the model when its performance on the validation dataset surpasses the previously saved model (or defined best metric). You can configure the save path and model name as needed. Additionally, the model parameters can be adjusted within the train_model function.

`test.py`  is used to test the saved DCT-UNet. You can configure the saved model path as needed.

## References
If you use the DCT-UNet model or find this repository helpful in your research, Please cite the following papers.
> Kim, G., Viswanathan, A.N., Bhatia, R., Landman, Y., Roumeliotis, M., Erickson, B., Schmidt, E.J. and Lee, J., 2024. Dual convolution-transformer UNet (DCT-UNet) for organs at risk and clinical target volume segmentation in MRI for cervical cancer brachytherapy. Physics in Medicine and Biology.
> Kim, G., Antaki, M., Schmidt, E.J., Roumeliotis, M., Viswanathan, A.N. and Lee, J., 2024, March. Intraoperative MRI-guided cervical cancer brachytherapy with automatic tissue segmentation using dual convolution-transformer network and real-time needle tracking. In Medical Imaging 2024: Image-Guided Procedures, Robotic Interventions, and Modeling (Vol. 12928, pp. 263-270). SPIE.


