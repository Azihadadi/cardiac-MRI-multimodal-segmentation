import torch
# from torchgeometry.losses.dice import DiceLoss

from learning.datasets import constants
from learning.datasets.DataSet2DSeg import DataSetACDC, DataSetCINEDE, DataSetEmidec, DataSetDE
from learning.models.architectures.UNet import UNet, FIUNet,FOUNet
from torch.utils.data import DataLoader
import torch.nn.functional as F

from learning.models.routines import apply_seg_model



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# load and split the dataset
fsave_im, fsave_scores = True, True  # whether to save the validation images and the results in .csv
dataset_name = constants.DATASET_CINEDE
save_volume = True
"""
    Test U-Net on the different dataset
"""
if dataset_name == constants.DATASET_ACDC:
    dataset_path = '../../data/ACDC'
    mono_input = True
    number_class = constants.ACDC_NUM_CLASS
    model = UNet(number_class).to(device)
    dataset2DSeg = DataSetACDC
    model_pth = "models/trained_models/unet_ACDC"
elif dataset_name == constants.DATASET_EMIDEC:
    dataset_path = '../../data/Emidec'
    mono_input = True
    number_class = constants.EMIDEC_NUM_CLASS
    model = UNet(number_class).to(device)
    dataset2DSeg = DataSetEmidec
    model_pth = "models/trained_models/unet_EMIDEC"
elif dataset_name == constants.DATASET_CINEDE:
    dataset_path = '../../data/CINEDE_272'
    mono_input = False
    number_class = constants.CINEDE_NUM_CLASS
    model = FOUNet(number_class).to(device)
    dataset2DSeg = DataSetCINEDE
    # model_pth = "models/trained_models/unet_CINEDE_272_input"
    model_pth = "models/trained_models/unet_CINEDE_272_output"
    # model_pth = "models/trained_models/unet_CINEDE_output_10_fold"

    # model_pth = "models/trained_models/unet_non_reg"
elif dataset_name == constants.DATASET_DE:
    dataset_path = '../../data/CINEDE_272'
    mono_input = True
    number_class = constants.CINEDE_NUM_CLASS # same as CINEDE dataset classes
    model = UNet(number_class).to(device)
    dataset2DSeg = DataSetDE
    # model_pth = "models/trained_models/unet_DE"
    model_pth = "models/trained_models/unet_DE_272_fold_1"

# create data loaders
valset = dataset2DSeg(dataset_path, 'test', (range(1,8), None))   # only for final tests

loss_fn = F.cross_entropy
valloader = DataLoader(valset, batch_size=1, shuffle=False)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

# load model : classical U-Net
print("Loading model")
model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))  # load weights

print("Start test phase ")
apply_seg_model(valloader, model, loss_fn, device, fsave_im, fsave_scores,dataset_name, mono_input, save_volume)
print("Done with testing!")
# Q10 : save results : flag saving image with contours, flag save dice scores with mean and std in .csv
# Q11: link res to patients (save image as a volume / png + nifti as option with the scores
# Q12 : add MAD, HD and cross validation
