"""
Training DCTUnet
"""

import torch
import os,glob
import numpy as np
import matplotlib.pyplot as plt
import monai.data
from monai.losses import DiceCELoss
import monai.transforms as mt
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from model import DCTUNet
from data_spliter import data_path_spliter
from data_loader import pre_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('training on torch.device -', device)

def train_model(
        img_base_path,
        model_base_path, 
        model_name,
        cls,
        crop_size,
        resample_spacing,
        epochs: int=2,
        lr: float=5*1e-4,
        batch_size: int=3,
        weight_decay: float=1e-2,
        best_metric: float=0.5,
        fold: int=5,
        fold_ind: int=1,
        random_state: int=7,
        val_test_ratio: int=0.8
        ):


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


    #### Parameters for train ####
    mul_batch_iter = 1 
    tr_check_intv = 5 
    val_check_epoch_intv = 1


    #### Data split####
    img_paths = glob.glob(img_base_path + '\\**\\image.nii', recursive=True)
    train_dicts, val_dicts, test_dicts =  data_path_spliter(img_paths, fold=fold, fold_ind=fold_ind, random_state=random_state, val_test_ratio=val_test_ratio) 


    #### Data load ####
    train_transform, val_transform, test_transform = pre_transforms(crop_size=crop_size, resample_spacing=resample_spacing)

    train_dataset = monai.data.Dataset(data=train_dicts, transform=train_transform)
    val_dataset = monai.data.Dataset(data=val_dicts, transform=val_transform)
    test_dataset = monai.data.Dataset(data=test_dicts, transform=test_transform)

    train_loader = monai.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = monai.data.DataLoader(val_dataset,batch_size=1)
    test_loader = monai.data.DataLoader(test_dataset,batch_size=1)


    #### Set tensorboard ####
    writer_dir = os.path.join(model_base_path,'logs', model_name)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=writer_dir)


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

    # Set optimizer and loss/metric function
    torch.backends.cudnn.benchmark = True
    loss_func = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.7, lambda_ce=0.3, batch=True)
    dice_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    post_softmax = mt.Activations(softmax=True)
    post_argmax = mt.AsDiscrete(argmax=True)
    post_pred = mt.AsDiscrete(argmax=True, to_onehot=out_channels)
    post_label = mt.AsDiscrete(to_onehot=out_channels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    #### Train model ####
    best_dir=[]
    epoch_result_total = np.zeros((1,4))
    epoch_result = np.zeros((1,4))
    mul_full_batch_no = int(len(train_loader) / mul_batch_iter)

    for epoch in range(epochs):
        print("-" * 20)
        print('Epoch: {} / {}'.format(epoch+1,epochs))
        

        # Train
        model.train()
        tr_loss = 0
        tr_dice = 0
        tr_check_no = 0
        mul_batch = 1
        optimizer.zero_grad() 
        for i, batch in enumerate(train_loader):

            inputs, labels = batch['image'].to(device), batch['label'].to(device) # data to device

            outputs = model(inputs) # run model

            loss = loss_func(outputs, labels)

            # Gradient accumulation
            if mul_batch <= mul_full_batch_no:
                (loss/mul_batch_iter).backward()
                tr_loss += loss.item()
            else:
                mul_batch_left = (len(train_loader) % mul_batch_iter)
                (loss/mul_batch_left).backward()
                tr_loss += loss.item()
            if (i+1) % mul_batch_iter == 0:
                optimizer.step() 
                optimizer.zero_grad()
                mul_batch += 1
            elif (i+1) == len(train_loader):
                optimizer.step() 
                optimizer.zero_grad()

            # Check dice score of train dataset
            if i % tr_check_intv == 0:
                tr_check_no += 1
                post_outputs = [post_pred(x) for x in monai.data.decollate_batch(outputs)]
                post_labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
                dice = np.nanmean(dice_func(post_outputs, post_labels).detach().cpu().numpy())
                tr_dice += dice
                dice_func.reset()

        tr_loss_mean = tr_loss/len(train_loader)
        tr_dice_mean = tr_dice/tr_check_no
        epoch_result[0,0] = tr_loss_mean
        epoch_result[0,1] = tr_dice_mean

        if (epoch + 1) % val_check_epoch_intv == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_dice = 0
                for i, batch in enumerate(val_loader):
                    
                    inputs, labels = batch['image'].to(device), batch['label'].to(device) # data to device

                    outputs = sliding_window_inference(inputs,crop_size,len(val_loader),model,overlap=0.7) # validate model

                    # Compute loss
                    loss = loss_func(outputs, labels)
                    val_loss += loss.item()

                    # Compute dice score
                    post_outputs = [post_pred(x) for x in monai.data.decollate_batch(outputs)]
                    post_labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
                    dice = np.nanmean(dice_func(post_outputs, post_labels).detach().cpu().numpy())
                    val_dice += dice
                    dice_func.reset()

        val_loss_mean = val_loss/len(val_loader)
        val_dice_mean = val_dice/len(val_loader)
        epoch_result[0,2] = val_loss_mean
        epoch_result[0,3] = val_dice_mean

        print('Average metric = tr_loss: {:.4f}, tr_dice:{:.4f}, val_loss:{:.4f}, val_dice:{:.4f}'.format(tr_loss_mean, tr_dice_mean, val_loss_mean, val_dice_mean))

        epoch_result_total = np.vstack((epoch_result_total,epoch_result))


        # Save best model
        if val_dice_mean > best_metric:
            best_metric = val_dice_mean
            tr_best_loss = tr_loss_mean
            tr_best_dice = tr_dice_mean
            val_best_loss = val_dice_mean
            best_dir = '\\' + model_name + '_epoch_{}.pt'.format(epoch+1)
            torch.save(model.state_dict(), model_base_path + best_dir)

        # write to summary
        loss_dict = {'train':tr_loss_mean, 'val':val_loss_mean}
        dice_dict = {'train':tr_dice_mean, 'val':val_dice_mean,}
        writer.add_scalars('Loss', loss_dict, epoch+1)
        writer.add_scalars('Dice', dice_dict, epoch+1)

        if scheduler is not None:
            scheduler.step()

    epoch_result_total = np.delete(epoch_result_total, (0), axis=0)

    print("-" * 20)
    print("Train is Finished!")
    if best_dir:
        print(best_dir)
        print('Best metric = tr_loss: {:.4f}, tr_dice:{:.4f}, val_loss:{:.4f}, val_dice: {:.4f}'.format(tr_best_loss, tr_best_dice, val_best_loss, best_metric))
    else:
        print('No saved model')

    #### Draw learning curve ####
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_result_total[:,0], label="tr_loss")
    plt.plot(epoch_result_total[:,2], label="val_loss")
    plt.subplot(1, 2, 2)
    plt.title("Epoch Average Dice")
    plt.xlabel("epoch")
    plt.ylabel("Dice")
    plt.plot(epoch_result_total[:,1], label="tr_dice")
    plt.plot(epoch_result_total[:,3], label="val_dice")
    plt.show()

if __name__ == '__main__':

    #### Set path ####
    """
    Edit path for your computer setting
    """
    model_base_path = 'D:\\DCTUNet\\Model_save' # Define file path for saving model
    model_name = 'Coarse_DCTUNet' # Define base name of saved model name
    img_base_path = 'D:\\DCTUNet\\Data' # Define file path for data


    train_model(img_base_path=img_base_path, model_base_path=model_base_path, model_name=model_name, cls=4, crop_size=[128,128,64], resample_spacing=[1.0,1.0,1.6])

