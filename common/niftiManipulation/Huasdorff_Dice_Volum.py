import cv2
import os
import numpy as np
import nibabel as nib
import common.constants as constants
from PIL import Image
from common import utils
from medpy.metric import hd
import seg_metrics.seg_metrics as sg

def dice_score(pred_slices,gt_slices):
    dsc = []
    for slice in range(pred_slices.shape[2]):
        pred = pred_slices[:,:,slice]
        gt = gt_slices[:,:,slice]

        volume_sum =  pred.sum() + gt.sum()

        if volume_sum == 0:
            return np.NaN
        volume_intersect = (gt & pred).sum()
        dsc.append(2 * volume_intersect / volume_sum)
    dsc_mean = np.mean(dsc)
    dsc_std = np.std(dsc)
    return dsc_mean, dsc_std

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

# save slices as a new nifti file with file_name in the path
def save_nifti(all_slices, path, file_name, header, is_myo):
    # save as nifti
    if not os.path.exists(path):
        os.makedirs(path)

    for slice_number in range(all_slices.shape[2]):
        slice = all_slices[:, :, slice_number]
        if is_myo:
            copy = slice.copy()
            copy[copy == 1] = 0  # create myo
        else:
            copy = slice.copy()
            copy[copy == 2] = 0
        all_slices[:, :, slice_number] = copy

    new_ref_nifti = nib.Nifti1Image(np.asarray(all_slices), np.eye(4), header=header.copy())
    nib.save(new_ref_nifti, os.path.join(path, file_name))

# convert slices to ndarray
def make_structures_arrays(path, height, width, is_myo):
    number_files = len(os.listdir(path))
    all_slices = np.ones((height, width, number_files), dtype=np.uint8)
    for slice_number in range(number_files):
        img = Image.open(path + "/slice_" + str(slice_number) + ".png")
        all_slices[:, :, slice_number] = img
        if is_myo:
            copy = all_slices[:, :, slice_number].copy()
            copy[copy == 1] = 0
            copy[copy == 2] = 1
        else:
            copy = all_slices[:, :, slice_number].copy()
            copy[copy == 2] = 0
        all_slices[:, :, slice_number] = copy
    return all_slices

res_hd =[]
res_dsc =[]
for patient_number in range (1,8):
    # PREDICT_PATH = os.path.join("E:\internship\code\multiple_MRI\multi-mri-seg\data\prediction_results\CINEDE\\output_non_reg_AB")
    PREDICT_PATH = os.path.join("E:\internship\code\multiple_MRI\multi-mri-seg\data\prediction_results\CINEDE\\x")
    # PREDICT_PATH = os.path.join("E:\internship\code\multiple_MRI\multi-mri-seg\data\prediction_results\DE\DE_AB")
    TESTING_PATH = os.path.join("E:\internship\code\multiple_MRI\multi-mri-seg\data\CINEDE_272\\testing\patient00"+str(patient_number))

    # prediction processing
    de_pred_case = os.path.join(PREDICT_PATH, "prediction_" + str(patient_number) + "_DE.nii.gz")
    de_pred = nib.load(de_pred_case)
    de_pred_data = de_pred.get_fdata()

    # GT processing
    de_gt_case = os.path.join(TESTING_PATH, "Case_00" + str(patient_number) + "_gt.nii.gz")
    de_gt = nib.load(de_gt_case)
    de_gt_data = de_gt.get_fdata()

    header = de_gt.header
    voxel_space = header['pixdim'][1:4]

    # load the slices
    (de_pred_H, de_pred_W) = load_slices(de_pred_data, constants.DE_MASK_SAVE_PATH)
    (de_gt_H, de_gt_W) = load_slices(de_gt_data, constants.CINE_MASK_SAVE_PATH)

    # decompose structures
    de_pred_slices = make_structures_arrays(constants.DE_MASK_SAVE_PATH, de_pred_H, de_pred_W, True)
    de_gt_slices = make_structures_arrays(constants.CINE_MASK_SAVE_PATH, de_gt_H, de_gt_W, True)

    de_pred_slices = de_pred_slices.transpose((1, 0, -1))
    de_gt_slices = de_gt_slices.transpose((1, 0, -1))

    hd_distance = hd(de_pred_slices, de_gt_slices, voxel_space)
    print("HD: " + str(hd_distance))
    dsc_mean, dsc_std = dice_score(de_pred_slices, de_gt_slices)
    print("DSC: " + str(dsc_mean))

    res_hd.append(hd_distance)
    res_dsc.append(dsc_mean)
    utils.remove_contents(constants.DE_MASK_SAVE_PATH)
    utils.remove_contents(constants.CINE_MASK_SAVE_PATH)


dsc_mean = np.mean(res_dsc)
dsc_std = np.std(res_dsc)

huasdorff_mean = np.mean(res_hd)
huasdorff_std = np.std(res_hd)

print("DSC mean: " + str(dsc_mean))
print("DSC std: " + str(dsc_std))

print("HD mean: " + str(huasdorff_mean))
print("HD std: " + str(huasdorff_std))

# labels = [0, 1, 2]
# res =[]
# for patient_number in range (1,8):
#     gdth_path = 'E:\internship\code\multiple_MRI\multi-mri-seg\data\CINEDE_272\\testing\patient00'+ str(patient_number) + '\Case_00'+ str(patient_number) + '_gt.nii.gz'
#     # pred_path = 'E:\internship\code\multiple_MRI\multi-mri-seg\data\prediction_results\CINEDE\\output_final_non_reg\prediction_'+ str(patient_number) + '_DE.nii.gz'
#     pred_path = 'E:\internship\code\multiple_MRI\multi-mri-seg\data\prediction_results\DE\\prediction_'+ str(patient_number) + '_DE.nii.gz'
#     csv_file = 'metrics.csv'
#
#     metrics = sg.write_metrics(labels=labels[1:],  # exclude background
#                       gdth_path=gdth_path,
#                       pred_path=pred_path,csv_file=csv_file)
#     res.append(metrics['dice'])
#
# print(res)
# huasdorff_mean = np.mean(res)
# huasdorff_std = np.std(res)
# print(huasdorff_mean)
# print(huasdorff_std)