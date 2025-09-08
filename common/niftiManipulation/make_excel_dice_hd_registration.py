import os
import numpy as np
import cv2
import nibabel as nib
import common.constants as constant
from PIL import Image

from common import utils

# extract and load slices
def load_slices(data, path):
    # transpose to convert x,y to height,width
    data = data.transpose((1, 0, -1))
    for slice_number in range(data.shape[2]):
        real_img = data[:, :, slice_number]
        img_fullname = path + "/slice_" + str(slice_number) + ".png"
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(img_fullname, real_img)
    return real_img.shape

def compute_statestics(cine_slices,de_slices,cine_patient,sheet_name):
    for slice_number in range(cine_slices.shape[2]):
        dice_score = utils.compute_avg_dice_score(cine_slices[:,:,slice_number], de_slices[:,:,slice_number])
        hausdorff = utils.compute_avg_hausdorff(cine_slices[:,:,slice_number], de_slices[:,:,slice_number])
        print(dice_score)
        print(hausdorff)
        utils.get_dice_hasudorff_distances(dice_score,hausdorff,cine_patient[-3:] + "-" + str(slice_number),sheet_name)

# convert slices to ndarray
def make_slices_arrays(path, height, width):
    number_files = len(os.listdir(path))
    all_slices = np.ones((height, width, number_files), dtype=np.uint8)
    for slice_number in range(number_files):
        img = Image.open(path + "/slice_" + str(slice_number) + ".png")
        all_slices[:, :, slice_number] = img
    return all_slices

# process the cases
cine_main_path = constant.CINE_ADAPTED_DATA_PATH_ROI
de_main_path = constant.DE_ADAPTED_DATA_PATH_ROI
sheet_name = "ROI"
cine_nifti_list = sorted(os.listdir(cine_main_path))
de_nifti_list = sorted(os.listdir(de_main_path))

for case_num in range(len(cine_nifti_list)):
    # Remove the temporary Slices
    utils.remove_contents(constant.CINE_MASK_SAVE_PATH)
    utils.remove_contents(constant.DE_MASK_SAVE_PATH)
    cine_folder = cine_nifti_list[case_num]
    de_folder = de_nifti_list[case_num]
    if cine_folder =="patient107":
        cine_case = os.path.join(cine_main_path,cine_folder,cine_folder + "_frame01_adapted_gt.nii.gz")
        de_case = os.path.join(de_main_path,de_folder,"Contours" , "Case_P" + de_folder[-3:] +"_adapted_gt.nii.gz")

        cine_gt = nib.load(cine_case)
        cine_gt_data = cine_gt.get_fdata()

        de_gt = nib.load(de_case)
        de_gt_data = de_gt.get_fdata()
        #
        # # load the slices
        (cine_gt_H, cine_gt_W) = load_slices(cine_gt_data, constant.CINE_MASK_SAVE_PATH)
        (de_gt_H, de_gt_W) = load_slices(de_gt_data, constant.DE_MASK_SAVE_PATH)

        cine_gt_slices = make_slices_arrays(constant.CINE_MASK_SAVE_PATH, cine_gt_H, cine_gt_W)
        de_gt_slices = make_slices_arrays(constant.DE_MASK_SAVE_PATH, de_gt_H, de_gt_W)

        cine_gt_slices = cine_gt_slices.transpose((1, 0, -1))
        de_gt_slices = de_gt_slices.transpose((1, 0, -1))

        compute_statestics(cine_gt_slices,de_gt_slices,cine_folder,sheet_name)



