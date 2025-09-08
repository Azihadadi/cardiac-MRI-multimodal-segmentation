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

        # Normalization
        normalized_img = np.zeros((data.shape[0], data.shape[1]))
        normalized_img = cv2.normalize(real_img, normalized_img, 0, 255, cv2.NORM_MINMAX)
        # Make a directory to save the slices if save mode is TRUE
        if not os.path.exists(path):
            os.makedirs(path)
        if is_gt:
            cv2.imwrite(img_fullname, real_img)
        else:
            cv2.imwrite(img_fullname, normalized_img)
    return real_img.shape

# save slices as a new nifti file with file_name in the path
def save_nifti(all_slices, path, file_name, is_gt, header, image_type):
    # save as nifti
    if not os.path.exists(path):
        os.makedirs(path)
    if image_type == constant.CINE_TYPE:
        file_suffix = '_frame01_gt.nii.gz' if is_gt else ".nii.gz"
    elif image_type == constant.DE_TYPE:
        file_suffix = '.nii.gz'

    for slice_number in range(all_slices.shape[2]):
        slice = all_slices[:, :, slice_number]
        slice[slice == 1] = 3
        slice[slice == 2] = 1
        slice[slice == 3] = 2
        # all_slices[:, :, slice_number] = slice
        normalized_img = np.zeros((all_slices.shape[0], all_slices.shape[1]))
        normalized_img = cv2.normalize(all_slices[:, :, slice_number], normalized_img, 0, 2, cv2.NORM_MINMAX)
        all_slices[:, :, slice_number] = normalized_img

    new_ref_nifti = nib.Nifti1Image(np.asarray(all_slices), np.eye(4), header=header.copy())
    nib.save(new_ref_nifti, os.path.join(path, file_name + file_suffix))

# convert slices to ndarray
def make_slices_arrays(path, height, width):
    number_files = len(os.listdir(path))
    all_slices = np.ones((height, width, number_files), dtype=np.uint8)
    for slice_number in range(number_files):
        img = Image.open(path + "/slice_" + str(slice_number) + ".png")
        all_slices[:, :, slice_number] = img
    return all_slices

# process the cases
for cine_number in range(177,179):
    # ground truth processing
    cine_gt_case = os.path.join("patient" + str(cine_number), "patient" + str(cine_number) + "_frame01_gt.nii.gz")
    # cine_gt_case = os.path.join("patient" + str(cine_number).zfill(3), "patient" + str(cine_number).zfill(3) + "_gt.nii.gz")
    # cine_gt_case = os.path.join("patient" + str(cine_number).zfill(3), "Case_" + str(cine_number).zfill(3) + "_gt.nii.gz")


    # CINE_MRI GT
    ROOT = "E:\internship\code\multiple_MRI\multi-mri-seg\data"
    # CINEDE_RG = os.path.join(ROOT, 'CINEDE_RG/training')
    ACDC = os.path.join(ROOT, 'ACDC/training')
    cine_gt = nib.load(os.path.join(ACDC, cine_gt_case))
    cine_gt_data = cine_gt.get_fdata()

    # load the slices
    (cine_gt_H, cine_gt_W) = load_slices(cine_gt_data, constant.CINE_MASK_SAVE_PATH,
                                         constant.IS_GT)

    # define the final path to save the new nifti files
    cine_final_path = os.path.join(ACDC, "patient" + str(cine_number).zfill(3))

    cine_gt_slices = make_slices_arrays(constant.CINE_MASK_SAVE_PATH, cine_gt_H, cine_gt_W)

    cine_gt_slices = cine_gt_slices.transpose((1, 0, -1))
    # save the new nifti files
    save_nifti(cine_gt_slices, cine_final_path, cine_gt_case.split(os.path.sep)[0] ,
               constant.IS_GT, cine_gt.header, constant.CINE_TYPE)

    # save_nifti(cine_gt_slices, cine_final_path, "Case_" + str(cine_number).zfill(3) ,
    #            constant.IS_GT, cine_gt.header, constant.CINE_TYPE)

    utils.remove_contents(constant.CINE_MASK_SAVE_PATH)
