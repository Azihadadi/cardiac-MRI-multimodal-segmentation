import torch
from learning.datasets import constants
from learning.datasets.DataSet2DSeg import DataSetACDC, DataSetEmidec, DataSetCINEDE, DataSetDE
from learning.models.architectures.network import R2AttU_Net
from learning.models.architectures.UNet import UNet, FIUNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
from learning.models.routines import train_seg_model, apply_seg_model
import matplotlib.pyplot as plt
import numpy as np
import copy

"""
    Train U-Net on the different datasets
"""

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# load and split the dataset
ftrain, fval, ftest = True, True, False  # whether to train, validate, test or not

# setup the dataset specifications
dataset_name = constants.DATASET_DE

if dataset_name == constants.DATASET_ACDC:
    dataset_path = '../../data/ACDC'
    dataset2DSeg = DataSetACDC
    total_cases = 100  # default value 100
    input_shape = (1, 256, 256)  # input shape for summary model
    is_mono_input = True
    number_class = constants.ACDC_NUM_CLASS
    model_type = UNet(number_class).to(device)

elif dataset_name == constants.DATASET_EMIDEC:
    dataset_path = '../../data/Emidec'
    dataset2DSeg = DataSetEmidec
    total_cases = 100  # default value 100
    input_shape = (1, 256, 256)  # input shape for summary model
    is_mono_input = True
    number_class = constants.EMIDEC_NUM_CLASS
    model_type = UNet(number_class).to(device)

elif dataset_name == constants.DATASET_CINEDE:
    dataset_path = '../../data/CINEDE'
    dataset2DSeg = DataSetCINEDE
    total_cases = 4  # default value 76
    input_shape = [(1, 256, 256), (1, 256, 256)]  # input shape for summary model
    is_mono_input = False
    number_class = constants.CINEDE_NUM_CLASS
    model_type = FIUNet(number_class).to(device)  # or FOUNet

elif dataset_name == constants.DATASET_DE:
    dataset_path = '../../data/CINEDE_RG'
    dataset2DSeg = DataSetDE
    total_cases = 68  # default value 76
    input_shape = (1, 256, 256)  # input shape for summary model
    is_mono_input = True
    number_class = constants.CINEDE_NUM_CLASS # same as CINEDE dataset classes
    model_type = R2AttU_Net(number_class).to(device)

k_folds = 2
# For fold results
results = {}
# Set fixed random number seed
torch.manual_seed(42)

# summary(model_type, input_shape)
# Define the K-fold Cross Validator manually
for fold in range(k_folds):
    print(f'FOLD {fold}')
    print('--------------------------------')
    split_size = total_cases // k_folds
    # Define start and end limitation for validation
    start_val = (fold * split_size) + 1
    end_val = start_val + split_size
    # Define data loaders for training and validation data in this fold
    trainset = dataset2DSeg(dataset_path, 'train',
                            (range(1, start_val), range(end_val, total_cases + 1)))  # training has 2 parts of splitting
    valset = dataset2DSeg(dataset_path, 'val', (range(start_val, end_val), None))  # validation has 1 part of splitting
    model = model_type #create new model based on type

    batch_size = 10
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=30, shuffle=True)
    # Training hyper-parameters
    loss_fn = F.cross_entropy
    # loss_fn = DiceLoss(use_background=True)
    learning_rate = 5e-4
    learning_rate_decay = 0.98
    L2_regular = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_regular)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,
                                                gamma=learning_rate_decay)  # learning rate decay scheduler

    epochs = 2
    model_pth = f"models/trained_models/unet_fold_{fold}"  # customize name later

    # checkpoint
    best_loss = float('inf')
    best_epoch = 0

    # early stopping
    stop_training = False
    stopped_epoch = 0
    wait = 0
    patience = 10
    # Training loop
    if ftrain is True:
        print("Beginning training \n")
        train_history_loss = []
        val_history_loss = []
        for t in range(epochs):
            # early stopping
            if stop_training:
                break

            print(f"Epoch {t + 1}\n-------------------------------")
            train_loss = train_seg_model(trainloader, model, loss_fn, optimizer, device, is_mono_input)
            train_history_loss.append(train_loss)
            val_loss = apply_seg_model(valloader, model, loss_fn, device, constants.NOT_SAVE_IMAGE,
                                       constants.NOT_SAVE_SCORE, dataset_name, is_mono_input)
            val_history_loss.append(val_loss)
            # checkpoint
            if val_loss < best_loss:
                best_model = copy.deepcopy(model)
                best_epoch = t + 1
                stopped_epoch = t + 1
                best_loss = val_loss
                wait = 1
            else:
                if wait >= patience:
                    stopped_epoch = t + 1
                    stop_training = True
                wait += 1
            scheduler.step()

    else:
        # model = FUNet().to(device)
        print("Loading model")
        model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))  # load weights

    if fval is True:
        print("Start test phase ")
        fold_val_loss = apply_seg_model(valloader, best_model, loss_fn, device, constants.NOT_SAVE_IMAGE,
                                        constants.IS_SAVE_SCORE, dataset_name, is_mono_input)

        plot_file_name = "results/loss_fold" + f"_{fold}_{best_epoch:03d}_{best_loss:.2f}.png"  # plotting the loss
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, stopped_epoch), train_history_loss, label="train_loss")
        plt.plot(np.arange(0, stopped_epoch), val_history_loss, label="validation_loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(plot_file_name)

        results[fold] = fold_val_loss
        print("Done with testing!")

    model_pth = model_pth + f'_{best_epoch:03d}_{best_loss:.2f}'
    torch.save(best_model.state_dict(), model_pth)

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')  # Print fold results
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average Fold Loss: {sum / len(results.items())} %\n')

    print("Done with training ! Saving model to :", model_pth, ", Checkpoint of epoch :", str(best_epoch))
    if stopped_epoch > 0:
        print('\nTerminated Training for Early Stopping at Epoch %04i' %
              (stopped_epoch))

# later : add saving of the training and validation losses, display images during training, HD
# Fix random seed, make hp clear (initialization, weight regularization, etc...), early stopping, selection on val
# Q13 : make a job -> UB
# Q14 : test azadeh mesocenter + show her the code and how to run it

# Azadeh : try to improve results (show her pytorch library, layers, talk hyperparameters, feature size, adjust split etc...)
# Look into nonewnet and acdc other proposed architectures
# -> 2 weeks
# Then same with Emidec
# No fusion at this stage
# Me : load from hdf5 or other, mask rcnn on acdc and emidec + bayesian optimization of classic unet
# add pre-post-processing
