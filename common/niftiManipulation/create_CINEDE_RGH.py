import os
import numpy as np
import cv2
import nibabel as nib
import common.constants as constant
from PIL import Image

from common import utils


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
def save_nifti(all_slices, path, file_name, header):
    # save as nifti
    if not os.path.exists(path):
        os.makedirs(path)
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

ROOT = "E:\internship\code\multiple_MRI\multi-mri-seg\data"
CINEDE_RGH = os.path.join(ROOT, 'CINEDE_RGH/training')
for num in range(221,229):

    cine_case = os.path.join(ROOT, "ACDC_adapted_roi_eh_dct", "training" , "patient" + str(num-50).zfill(3), "patient" + str(num-50) + "_frame01_adapted.nii.gz")
    cine_gt_case = os.path.join(ROOT, "ACDC_adapted_roi_eh_dct", "training" ,"patient" + str(num-50).zfill(3), "patient" + str(num-50) + "_frame01_adapted_gt.nii.gz")

    de_case = os.path.join(ROOT, "Emidec_adapted_roi_eh_dct", "Case_" + str(num), "Images", "Case_" + str(num) + "_adapted.nii.gz")
    de_gt_case = os.path.join(ROOT, "Emidec_adapted_roi_eh_dct", "Case_" + str(num), "Contours", "Case_P" + str(num) + "_adapted_gt.nii.gz")

    # DE_DCT_MRI
    cine_img = nib.load(cine_case)
    cine_data = cine_img.get_fdata()

    cine_gt = nib.load(cine_gt_case)
    cine_gt_data = cine_gt.get_fdata()

    de_img = nib.load(de_case)
    de_data = de_img.get_fdata()

    de_gt = nib.load(de_gt_case)
    de_gt_data = de_gt.get_fdata()

    #loading
    (cine_H, cine_W) = load_slices(cine_data, constant.CINE_SAVE_PATH, constant.IS_NOT_GT)
    (cine_gt_H, cine_gt_W) = load_slices(cine_gt_data, constant.CINE_MASK_SAVE_PATH, constant.IS_GT)
    (de_H, de_W) = load_slices(de_data, constant.DE_SAVE_PATH, constant.IS_NOT_GT)
    (de_gt_H, de_gt_W) = load_slices(de_gt_data, constant.DE_MASK_SAVE_PATH, constant.IS_GT)

    sorted_files = sorted(os.listdir(CINEDE_RGH))
    last_file_name = (sorted_files[-1])[7:].lstrip('0')
    final_path = os.path.join(CINEDE_RGH, "patient" + str(int(last_file_name) + 1).zfill(3))

    cine_real_slices = make_slices_arrays(constant.CINE_SAVE_PATH, cine_H, cine_W)
    cine_gt_slices = make_slices_arrays(constant.CINE_MASK_SAVE_PATH, cine_gt_H, cine_gt_W)
    de_real_slices = make_slices_arrays(constant.DE_SAVE_PATH, de_H, de_W)
    de_gt_slices = make_slices_arrays(constant.DE_MASK_SAVE_PATH, de_gt_H, de_gt_W)

    cine_real_slices = cine_real_slices.transpose((1, 0, -1))
    cine_gt_slices = cine_gt_slices.transpose((1, 0, -1))
    de_real_slices = de_real_slices.transpose((1, 0, -1))
    de_gt_slices = de_gt_slices.transpose((1, 0, -1))


    # # ground truth processing
    new_cine_name = os.path.join("patient" + str(int(last_file_name) + 1).zfill(3) + ".nii.gz")
    new_cine_gt_name = os.path.join("patient" + str(int(last_file_name) + 1).zfill(3) + "_gt.nii.gz")
    new_de_name = os.path.join("Case_" + str(int(last_file_name) + 1).zfill(3) + ".nii.gz")
    new_de_gt_name = os.path.join("Case_" + str(int(last_file_name) + 1).zfill(3) + "_gt.nii.gz")


    # save the new nifti files
    save_nifti(cine_real_slices, final_path, new_cine_name,  cine_img.header)
    save_nifti(cine_gt_slices, final_path, new_cine_gt_name, cine_gt.header)
    save_nifti(de_real_slices, final_path, new_de_name,  de_img.header)
    save_nifti(de_gt_slices, final_path, new_de_gt_name, de_gt.header)
    # Remove the temporary Slices
    utils.remove_contents(constant.CINE_SAVE_PATH)
    utils.remove_contents(constant.CINE_MASK_SAVE_PATH)
    utils.remove_contents(constant.DE_SAVE_PATH)
    utils.remove_contents(constant.DE_MASK_SAVE_PATH)


