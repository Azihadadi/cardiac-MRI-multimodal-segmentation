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


# sorted_files = sorted(os.listdir(os.path.join(constant.ROOT, 'CINEDE_RGH/training')))
# last_file_name = (sorted_files[-1])
# last_number = last_file_name[7:].lstrip('0')
# number = int(last_number) + 1

# number of cases
for number in range(1,77):
    # number = 1

    # # target slices numbers
    # target_slices = [0,8]

    # real image processing
    cine_case = os.path.join(constant.CINEDE_RGH_DATA_PATH,"patient" + str(number).zfill(3), "patient" + str(number).zfill(3) + ".nii.gz")
    de_case = os.path.join(constant.CINEDE_RGH_DATA_PATH,"patient" + str(number).zfill(3), "Case_" + str(number).zfill(3) + ".nii.gz")


    # CINE_MRI
    cine_img = nib.load(os.path.join(constant.CINE_DATA_PATH, cine_case))
    cine_data = cine_img.get_fdata()

    # DE_MRI
    de_img = nib.load(os.path.join(constant.DE_DATA_PATH, de_case))
    de_data = de_img.get_fdata()


    # load the slices
    (cine_H, cine_W) = load_slices(cine_data, constant.CINE_SAVE_PATH, constant.IS_NOT_GT)
    (de_H, de_W) = load_slices(de_data, constant.DE_SAVE_PATH, constant.IS_NOT_GT)


    # ground truth processing
    cine_gt_case = os.path.join(constant.CINEDE_RGH_DATA_PATH,"patient" + str(number).zfill(3), "patient" + str(number).zfill(3) + "_gt.nii.gz")
    de_gt_case = os.path.join(constant.CINEDE_RGH_DATA_PATH,"patient" + str(number).zfill(3), "Case_" + str(number).zfill(3) + "_gt.nii.gz")


    # CINE_MRI GT
    cine_gt = nib.load(os.path.join(constant.CINE_DATA_PATH, cine_gt_case))
    cine_gt_data = cine_gt.get_fdata()

    # DE_MRI GT
    de_gt = nib.load(os.path.join(constant.DE_GT_DATA_PATH, de_gt_case))
    de_gt_data = de_gt.get_fdata()


    # load the slices
    (cine_gt_H, cine_gt_W) = load_slices(cine_gt_data, constant.CINE_MASK_SAVE_PATH,
                                         constant.IS_GT)
    (de_gt_H, de_gt_W) = load_slices(de_gt_data, constant.DE_MASK_SAVE_PATH, constant.IS_GT)

    # define the final path to save the new nifti files
    final_path = os.path.join(constant.CINEDE_RGH_AB_DATA_PATH, "patient" + str(number).zfill(3))

    cine_real_slices = make_slices_arrays(constant.CINE_SAVE_PATH, cine_H, cine_W)
    de_real_slices = make_slices_arrays(constant.DE_SAVE_PATH, de_H, de_W)
    cine_gt_slices = make_slices_arrays(constant.CINE_MASK_SAVE_PATH, cine_gt_H, cine_gt_W)
    de_gt_slices = make_slices_arrays(constant.DE_MASK_SAVE_PATH, de_gt_H, de_gt_W)

    cine_real_slices = cine_real_slices.transpose((1, 0, -1))
    de_real_slices = de_real_slices.transpose((1, 0, -1))
    cine_gt_slices = cine_gt_slices.transpose((1, 0, -1))
    de_gt_slices = de_gt_slices.transpose((1, 0, -1))

    # target slices numbers
    target_slices = [0,cine_real_slices.shape[2]-1]
    # remove slices
    cine_real_slices = np.delete(cine_real_slices, target_slices, axis=2)
    de_real_slices = np.delete(de_real_slices, target_slices, axis=2)
    cine_gt_slices = np.delete(cine_gt_slices, target_slices, axis=2)
    de_gt_slices = np.delete(de_gt_slices, target_slices, axis=2)


    # save the new nifti files
    # print ((cine_img.header))
    save_nifti(cine_real_slices, final_path, "patient" + str(number).zfill(3),
               constant.IS_NOT_GT, cine_img.header, constant.CINE_TYPE)
    save_nifti(de_real_slices, final_path, "Case_" + str(number).zfill(3), constant.IS_NOT_GT, de_img.header,
               constant.DE_TYPE)

    save_nifti(cine_gt_slices, final_path, "patient" + str(number).zfill(3),
               constant.IS_GT, cine_gt.header, constant.CINE_TYPE)
    save_nifti(de_gt_slices, final_path, "Case_" + str(number).zfill(3), constant.IS_GT, de_gt.header, constant.DE_TYPE)

    # Remove the temporary Slices
    utils.remove_contents(constant.CINE_SAVE_PATH)
    utils.remove_contents(constant.DE_SAVE_PATH)
    utils.remove_contents(constant.CINE_MASK_SAVE_PATH)
    utils.remove_contents(constant.DE_MASK_SAVE_PATH)
