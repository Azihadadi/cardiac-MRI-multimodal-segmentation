import os
from builtins import range

import numpy as np
import cv2
import nibabel as nib
from skimage.util import random_noise

import common.constants as constant
from PIL import Image

from common import utils
import shutil

def rotate_img(image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    # rotate our image by 45 degrees
    M = cv2.getRotationMatrix2D((cX, cY), 5, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return  rotated

def shift_img(image):
    M = np.float32([[1, 0, 20], [0, 1, 20]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return  shifted

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
    file_suffix = '_gt.nii.gz' if is_gt else ".nii.gz"
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


# number of cases
for number in range(1,69):
    # real image processing
    cine_case = os.path.join(constant.CINEDE_RGH_BLUR_DATA_PATH, "patient" + str(number).zfill(3),
                             "patient" + str(number).zfill(3) + ".nii.gz")
    de_case = os.path.join(constant.CINEDE_RGH_BLUR_DATA_PATH, "patient" + str(number).zfill(3),
                           "Case_" + str(number).zfill(3) + ".nii.gz")

    # ground truth processing
    cine_gt_case = os.path.join(constant.CINEDE_RGH_BLUR_DATA_PATH,"patient" + str(number).zfill(3), "patient" + str(number).zfill(3) + "_gt.nii.gz")
    de_gt_case = os.path.join(constant.CINEDE_RGH_BLUR_DATA_PATH,"patient" + str(number).zfill(3), "Case_" + str(number).zfill(3) + "_gt.nii.gz")

    # CINE_MRI
    cine_img = nib.load(os.path.join(constant.CINE_DATA_PATH, cine_case))
    cine_data = cine_img.get_fdata()

    # DE_MRI
    de_img = nib.load(os.path.join(constant.DE_DATA_PATH, de_case))
    de_data = de_img.get_fdata()
    #----------------------GT--------------------------------------------
    # CINE_MRI
    cine_gt_img = nib.load(os.path.join(constant.CINE_DATA_PATH, cine_gt_case))
    cine_gt_data = cine_gt_img.get_fdata()

    # DE_MRI
    de_gt_img = nib.load(os.path.join(constant.DE_DATA_PATH, de_gt_case))
    de_gt_data = de_gt_img.get_fdata()

    # load the slices
    (cine_H, cine_W) = load_slices(cine_data, constant.CINE_SAVE_PATH, constant.IS_NOT_GT)
    (de_H, de_W) = load_slices(de_data, constant.DE_SAVE_PATH, constant.IS_NOT_GT)

    (cine_gt_H, cine_gt_W) = load_slices(cine_gt_data, constant.CINE_MASK_SAVE_PATH, constant.IS_GT)
    (de_gt_H, de_gt_W) = load_slices(de_gt_data, constant.DE_MASK_SAVE_PATH, constant.IS_GT)

    # define the final path to save the new nifti files
    final_path = os.path.join(constant.CINEDE_RGH_BLUR_DATA_PATH, "patient" + str(number+204).zfill(3))

    cine_real_slices = make_slices_arrays(constant.CINE_SAVE_PATH, cine_H, cine_W)
    de_real_slices = make_slices_arrays(constant.DE_SAVE_PATH, de_H, de_W)

    cine_real_slices = cine_real_slices.transpose((1, 0, -1))
    de_real_slices = de_real_slices.transpose((1, 0, -1))

#GT
    cine_gt_slices = make_slices_arrays(constant.CINE_MASK_SAVE_PATH, cine_gt_H, cine_gt_W)
    de_gt_slices = make_slices_arrays(constant.DE_MASK_SAVE_PATH, de_gt_H, de_gt_W)

    cine_gt_slices = cine_gt_slices.transpose((1, 0, -1))
    de_gt_slices = de_gt_slices.transpose((1, 0, -1))


    for slice in range(cine_real_slices.shape[2]):
        # cine_real_slices[:, :, slice] = cv2.GaussianBlur(cine_real_slices[:, :, slice], (3, 3), 0)
        # de_real_slices[:, :, slice] = cv2.GaussianBlur(de_real_slices[:, :, slice], (3, 3), 0)


        cine_real_slices[:, :, slice] = rotate_img(cine_real_slices[:, :, slice])
        de_real_slices[:, :, slice] = rotate_img(de_real_slices[:, :, slice])
        cine_gt_slices[:, :, slice] = rotate_img(cine_gt_slices[:, :, slice])
        de_gt_slices[:, :, slice] = rotate_img(de_gt_slices[:, :, slice])

        # cine_real_slices[:, :, slice] = shift_img(cine_real_slices[:, :, slice])
        # de_real_slices[:, :, slice] = shift_img(de_real_slices[:, :, slice])
        # cine_gt_slices[:, :, slice] = shift_img(cine_gt_slices[:, :, slice])
        # de_gt_slices[:, :, slice] = shift_img(de_gt_slices[:, :, slice])
        # save the new nifti files
    save_nifti(cine_real_slices, final_path, "patient" + str(number+204).zfill(3),
               constant.IS_NOT_GT, cine_img.header, constant.CINE_TYPE)
    save_nifti(de_real_slices, final_path, "Case_" + str(number+204).zfill(3), constant.IS_NOT_GT, de_img.header,
               constant.DE_TYPE)
    #new GT
    save_nifti(cine_gt_slices, final_path, "patient" + str(number+204).zfill(3),
               constant.IS_GT, cine_img.header, constant.CINE_TYPE)
    save_nifti(de_gt_slices, final_path, "Case_" + str(number+204).zfill(3), constant.IS_GT, de_img.header,
               constant.DE_TYPE)


    #rename GT
    # new_cine_gt_case = os.path.join(constant.CINEDE_RGH_BLUR_DATA_PATH,"patient" + str(number+204).zfill(3), "patient" + str(number+204).zfill(3) + "_gt.nii.gz")
    # new_de_gt_case = os.path.join(constant.CINEDE_RGH_BLUR_DATA_PATH,"patient" + str(number+204).zfill(3), "Case_" + str(number+204).zfill(3) + "_gt.nii.gz")
    # shutil.copy(cine_gt_case, new_cine_gt_case)
    # shutil.copy(de_gt_case, new_de_gt_case)

    # Remove the temporary Slices
    utils.remove_contents(constant.CINE_SAVE_PATH)
    utils.remove_contents(constant.DE_SAVE_PATH)
    utils.remove_contents(constant.CINE_MASK_SAVE_PATH)
    utils.remove_contents(constant.DE_MASK_SAVE_PATH)

