import os
import numpy as np
import cv2
import nibabel as nib
import common.constants as constant
from PIL import Image

from common import utils

# extract and load slices
def load_slices(data, path, is_gt):
    # transpose to convert x,y to height,width
    data = data.transpose((1, 0, -1))
    for slice_number in range(data.shape[2]):
        real_img = data[:, :, slice_number]
        img_fullname = path + "/slice_" + str(slice_number) + ".png"
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(img_fullname, real_img)
    return real_img.shape

# save slices as a new nifti file with file_name in the path
def save_nifti(all_slices, path, file_name, is_gt, header, image_type):
    # save as nifti
    if not os.path.exists(path):
        os.makedirs(path)

    for slice_number in range(all_slices.shape[2]):
        slice = all_slices[:, :, slice_number]
        myo_copy = slice.copy()
        endo_copy = slice.copy()
        myo_copy[myo_copy == 1] = 0 # create myo
        endo_copy[endo_copy == 2] = 0


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        endo_copy = cv2.morphologyEx(endo_copy, cv2.MORPH_OPEN, kernel)

        final_slice = myo_copy+endo_copy
        all_slices[:, :, slice_number] = final_slice

    new_ref_nifti = nib.Nifti1Image(np.asarray(all_slices), np.eye(4), header=header.copy())
    nib.save(new_ref_nifti, os.path.join(path, file_name))

# convert slices to ndarray
def make_slices_arrays(path, height, width):
    number_files = len(os.listdir(path))
    all_slices = np.ones((height, width, number_files), dtype=np.uint8)
    for slice_number in range(number_files):
        img = Image.open(path + "/slice_" + str(slice_number) + ".png")
        all_slices[:, :, slice_number] = img
    return all_slices

CINEDE_152 = os.path.join("E:\internship\code\multiple_MRI\multi-mri-seg\data", 'CINEDE_RG/training')
# process the cases
for cine_number in range(1,76):
    # ground truth processing
    cine_gt_case = os.path.join(CINEDE_152,"patient" + str(cine_number).zfill(3), "patient" + str(cine_number).zfill(3) + "_gt.nii.gz")

    cine_gt = nib.load(cine_gt_case)
    cine_gt_data = cine_gt.get_fdata()

    # load the slices
    (cine_gt_H, cine_gt_W) = load_slices(cine_gt_data, constant.CINE_MASK_SAVE_PATH,
                                         constant.IS_GT)

    # define the final path to save the new nifti files
    cine_final_path = os.path.join(CINEDE_152,"patient" + str(cine_number).zfill(3))

    cine_gt_slices = make_slices_arrays(constant.CINE_MASK_SAVE_PATH, cine_gt_H, cine_gt_W)

    file_name = "patient" + str(cine_number).zfill(3) + "_gt.nii.gz"
    cine_gt_slices = cine_gt_slices.transpose((1, 0, -1))
    # save the new nifti files
    save_nifti(cine_gt_slices, cine_final_path, file_name ,
               constant.IS_GT, cine_gt.header, constant.CINE_TYPE)
    utils.remove_contents(constant.CINE_MASK_SAVE_PATH)
