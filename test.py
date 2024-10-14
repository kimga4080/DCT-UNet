"""
Testing the saved DCTUnet
"""

import torch
import os,glob
import numpy as np
import monai.data
import monai.transforms as mt
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from model import DCTUNet
from data_spliter import data_path_spliter
from data_loader import pre_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('training on torch.device -', device)

def test_model(
        img_base_path,
        model_base_path, 
        best_model,
        cls,
        crop_size,
        resample_spacing,
        fold: int=5,
        fold_ind: int=1,
        random_state: int=7,
        val_test_ratio: int=0.8
        ):


    #### Data split####
    img_paths = glob.glob(img_base_path + '\\**\\image.nii', recursive=True)
    _, _, test_dicts =  data_path_spliter(img_paths, fold=fold, fold_ind=fold_ind, random_state=random_state, val_test_ratio=val_test_ratio) 


    #### Data load ####
    _, _, test_transform = pre_transforms(crop_size=crop_size, resample_spacing=resample_spacing)
    test_dataset = monai.data.Dataset(data=test_dicts, transform=test_transform)
    test_loader = monai.data.DataLoader(test_dataset,batch_size=1)


    #### Load the model ####
    #### Parameters for model ####
    in_channels = 1
    out_channels = cls + 1
    drop_rate = 0.1
    attn_drop_rate = 0.1
    dropout_path_rate = 0.1
    depths = (2,2,2) 
    self_atts=["Local", "Local", "Local"]
    patch_size = (2,2,2)
    window_size = (4,4,4)
    feature_size = 12
    use_checkpoint = True
    drop_rate_conv = 0.1
    spatial_dims=3
    channels=(16, 32, 64, 128)
    strides=(2, 2, 2, 2)
    num_res_units=2


    #### Creat model ####
    # Load model
    model = DCTUNet(
        img_size=crop_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depths=depths,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        patch_size=patch_size,
        window_size=window_size,
        drop_rate_conv=drop_rate_conv,
        spatial_dims=spatial_dims,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        self_atts=self_atts,
        adn_ordering = "NAD",
    ).to(device)

    model.load_state_dict(torch.load(model_base_path + '\\' + best_model))
    model.to(device)
    model.eval()

    # Set metric function
    torch.backends.cudnn.benchmark = True
    dice_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    post_pred = mt.AsDiscrete(argmax=True, to_onehot=out_channels)
    post_label = mt.AsDiscrete(to_onehot=out_channels)

    test_result_each = []
    with torch.no_grad():
        test_dice = 0
        for i, batch in enumerate(test_loader):
            
            inputs, labels = batch['image'].to(device), batch['label'].to(device) # data to device

            outputs = sliding_window_inference(inputs,crop_size,len(test_loader),model,overlap=0.5) # test model

            # Compute dice score
            post_outputs = [post_pred(x) for x in monai.data.decollate_batch(outputs)]
            post_labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
            dice = np.nanmean(dice_func(post_outputs, post_labels).detach().cpu().numpy())
            test_dice += dice
            dice_func.reset()
            test_result_each.append(dice)

        test_dice_mean = test_dice/len(test_loader)  

    print("-" * 20)
    print("Test is Finished!")   
    print('Test result = test_dice:{:.4f}'.format(test_dice_mean))


if __name__ == '__main__':

    #### Set path ####
    """
    Edit path for your computer setting
    """
    model_base_path = 'D:\\DCTUNet\\Model_save' # Define file path for saving model
    best_model = 'Coarse_DCTUNet_epoch_2.pt' # Define the name of saved model to test
    img_base_path = 'D:\\DCTUNet\\Data' # Define file path for data

    test_model(img_base_path=img_base_path, model_base_path=model_base_path, best_model=best_model, cls=4, crop_size=[128,128,64], resample_spacing=[1.0,1.0,1.6])
