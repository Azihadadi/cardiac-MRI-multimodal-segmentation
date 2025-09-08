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
def save_nifti(all_slices, path, file_name, is_gt, header, image_type, is_dct):
    # save as nifti
    if not os.path.exists(path):
        os.makedirs(path)
    if not is_dct and image_type == constant.CINE_TYPE:
        file_suffix = '_gt.nii.gz' if is_gt else ".nii.gz"
    elif not is_dct and image_type == constant.DE_TYPE:
        file_suffix = '.nii.gz'

    if is_dct and image_type == constant.CINE_TYPE:
        file_suffix = '_adapted_gt.nii.gz' if is_gt else "_adapted.nii.gz"
    elif is_dct and image_type == constant.DE_TYPE:
        file_suffix = '_adapted_gt.nii.gz' if is_gt else "_adapted.nii.gz"

    new_ref_nifti = nib.Nifti1Image(np.asarray(all_slices), np.eye(4))
    # set the header values
    new_ref_nifti.header['pixdim'] = header['pixdim']
    new_ref_nifti.header['sizeof_hdr'] = header['sizeof_hdr']
    new_ref_nifti.header['data_type'] = header['data_type']
    new_ref_nifti.header['db_name'] = header['db_name']
    new_ref_nifti.header['extents'] = header['extents']
    new_ref_nifti.header['session_error'] = header['session_error']
    new_ref_nifti.header['regular'] = header['regular']
    new_ref_nifti.header['dim_info'] = header['dim_info']
    new_ref_nifti.header['intent_p1'] = header['intent_p1']
    new_ref_nifti.header['intent_p2'] = header['intent_p2']
    new_ref_nifti.header['intent_p3'] = header['intent_p3']
    new_ref_nifti.header['intent_code'] = header['intent_code']
    new_ref_nifti.header['datatype'] = header['datatype']
    new_ref_nifti.header['bitpix'] = header['bitpix']
    new_ref_nifti.header['slice_start'] = header['slice_start']
    #
    new_ref_nifti.header['vox_offset'] = header['vox_offset']
    new_ref_nifti.header['scl_slope'] = header['scl_slope']
    new_ref_nifti.header['scl_inter'] = header['scl_inter']
    new_ref_nifti.header['slice_end'] = header['slice_end']
    new_ref_nifti.header['slice_code'] = header['slice_code']
    new_ref_nifti.header['xyzt_units'] = header['xyzt_units']
    new_ref_nifti.header['cal_max'] = header['cal_max']
    new_ref_nifti.header['cal_min'] = header['cal_min']
    new_ref_nifti.header['slice_duration'] = header['slice_duration']
    new_ref_nifti.header['toffset'] = header['toffset']
    new_ref_nifti.header['glmax'] = header['glmax']
    new_ref_nifti.header['glmin'] = header['glmin']
    new_ref_nifti.header['descrip'] = header['descrip']
    new_ref_nifti.header['aux_file'] = header['aux_file']
    new_ref_nifti.header['qform_code'] = header['qform_code']
    # new_ref_nifti.header['sform_code'] = header['sform_code']
    new_ref_nifti.header['quatern_b'] = header['quatern_b']
    new_ref_nifti.header['quatern_c'] = header['quatern_c']
    new_ref_nifti.header['quatern_d'] = header['quatern_d']
    new_ref_nifti.header['qoffset_x'] = header['qoffset_x']
    new_ref_nifti.header['qoffset_y'] = header['qoffset_y']
    new_ref_nifti.header['qoffset_z'] = header['qoffset_z']
    new_ref_nifti.header['intent_name'] = header['intent_name']
    new_ref_nifti.header['magic'] = header['magic']
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
cine_number = 178
de_number = 228
# target slices numbers
cine_target_slices = [0,1]
de_target_slices = [0,1]
cine_gt_target_slices = [0,1]
de_gt_target_slices = [0,1]

#dct
cine_dct_target_slices = [0,1]
de_dct_target_slices = [0,1]
cine_dct_gt_target_slices = [0,1]
de_dct_gt_target_slices = [0,1]

# real image processing
cine_case = os.path.join("patient" + str(cine_number), "patient" + str(cine_number) + "_frame01.nii.gz")
de_case = os.path.join("Case_" + str(de_number), "Images", "Case_" + str(de_number) + ".nii.gz")

# dct
cine_dct_case = os.path.join("patient" + str(cine_number), "patient" + str(cine_number) + "_frame01_adapted.nii.gz")
de_dct_case = os.path.join("Case_" + str(de_number), "Images", "Case_" + str(de_number) + "_adapted.nii.gz")

# CINE_MRI
cine_img = nib.load(os.path.join(constant.CINE_DATA_PATH, cine_case))
cine_data = cine_img.get_fdata()

# DE_MRI
de_img = nib.load(os.path.join(constant.DE_DATA_PATH, de_case))
de_data = de_img.get_fdata()


# CINE_DCT_MRI
cine_dct_img = nib.load(os.path.join(constant.CINE_ADAPTED_DATA_PATH_ROI_DTC, cine_dct_case))
cine_dct_data = cine_dct_img.get_fdata()

# DE_DCT_MRI
de_dct_img = nib.load(os.path.join(constant.DE_ADAPTED_DATA_PATH_ROI_DCT, de_dct_case))
de_dct_data = de_dct_img.get_fdata()

# load the slices
(cine_H, cine_W) = load_slices(cine_data, constant.CINE_SAVE_PATH, constant.IS_NOT_GT)
(de_H, de_W) = load_slices(de_data, constant.DE_SAVE_PATH, constant.IS_NOT_GT)

#dct
(cine_dct_H, cine_dct_W) = load_slices(cine_dct_data, constant.CINE_DCT_SAVE_PATH, constant.IS_NOT_GT)
(de_dct_H, de_dct_W) = load_slices(de_dct_data, constant.DE_DCT_SAVE_PATH, constant.IS_NOT_GT)

# ground truth processing
cine_gt_case = os.path.join("patient" + str(cine_number), "patient" + str(cine_number) + "_frame01_gt.nii.gz")
de_gt_case = os.path.join("Case_" + str(de_number), "Contours", "Case_P" + str(de_number) + ".nii.gz")

# dct
cine_dct_gt_case = os.path.join("patient" + str(cine_number), "patient" + str(cine_number) + "_frame01_adapted_gt.nii.gz")
de_dct_gt_case = os.path.join("Case_" + str(de_number), "Contours", "Case_P" + str(de_number) + "_adapted_gt.nii.gz")



# CINE_MRI GT
cine_gt = nib.load(os.path.join(constant.CINE_DATA_PATH, cine_gt_case))
cine_gt_data = cine_gt.get_fdata()

# DE_MRI GT
de_gt = nib.load(os.path.join(constant.DE_GT_DATA_PATH, de_gt_case))
de_gt_data = de_gt.get_fdata()

# CINE_DCT_MRI GT
cine_dct_gt = nib.load(os.path.join(constant.CINE_ADAPTED_DATA_PATH_ROI_DTC, cine_dct_gt_case))
cine_dct_gt_data = cine_dct_gt.get_fdata()

# DE_DCT_MRI GT
de_dct_gt = nib.load(os.path.join(constant.DE_ADAPTED_DATA_PATH_ROI_DCT, de_dct_gt_case))
de_dct_gt_data = de_dct_gt.get_fdata()



# load the slices
(cine_gt_H, cine_gt_W) = load_slices(cine_gt_data, constant.CINE_MASK_SAVE_PATH,
                                     constant.IS_GT)
(de_gt_H, de_gt_W) = load_slices(de_gt_data, constant.DE_MASK_SAVE_PATH, constant.IS_GT)

#dct
(cine_dct_gt_H, cine_dct_gt_W) = load_slices(cine_dct_gt_data, constant.CINE_DCT_MASK_SAVE_PATH,
                                     constant.IS_GT)
(de_dct_gt_H, de_dct_gt_W) = load_slices(de_dct_gt_data, constant.DE_DCT_MASK_SAVE_PATH, constant.IS_GT)

# define the final path to save the new nifti files
cine_final_path = os.path.join(constant.CINE_DATA_PATH, "patient" + str(cine_number))
de_real_final_path = os.path.join(constant.DE_DATA_PATH, "Case_" + str(de_number), "Images")
de_gt_final_path = os.path.join(constant.DE_GT_DATA_PATH, "Case_" + str(de_number), "Contours")


#dct
cine_dct_final_path = os.path.join(constant.CINE_ADAPTED_DATA_PATH_ROI_DTC, "patient" + str(cine_number))
de_dct_real_final_path = os.path.join(constant.DE_ADAPTED_DATA_PATH_ROI_DCT, "Case_" + str(de_number), "Images")
de_dct_gt_final_path = os.path.join(constant.DE_ADAPTED_DATA_PATH_ROI_DCT, "Case_" + str(de_number), "Contours")

cine_real_slices = make_slices_arrays(constant.CINE_SAVE_PATH, cine_H, cine_W)
de_real_slices = make_slices_arrays(constant.DE_SAVE_PATH, de_H, de_W)
cine_gt_slices = make_slices_arrays(constant.CINE_MASK_SAVE_PATH, cine_gt_H, cine_gt_W)
de_gt_slices = make_slices_arrays(constant.DE_MASK_SAVE_PATH, de_gt_H, de_gt_W)

# dct
cine_dct_real_slices = make_slices_arrays(constant.CINE_DCT_SAVE_PATH, cine_dct_H, cine_dct_W)
de_dct_real_slices = make_slices_arrays(constant.DE_DCT_SAVE_PATH, de_dct_H, de_dct_W)
cine_dct_gt_slices = make_slices_arrays(constant.CINE_DCT_MASK_SAVE_PATH, cine_dct_gt_H, cine_dct_gt_W)
de_dct_gt_slices = make_slices_arrays(constant.DE_DCT_MASK_SAVE_PATH, de_dct_gt_H, de_dct_gt_W)

cine_real_slices = cine_real_slices.transpose((1, 0, -1))
de_real_slices = de_real_slices.transpose((1, 0, -1))
cine_gt_slices = cine_gt_slices.transpose((1, 0, -1))
de_gt_slices = de_gt_slices.transpose((1, 0, -1))

# dct
cine_dct_real_slices = cine_dct_real_slices.transpose((1, 0, -1))
de_dct_real_slices = de_dct_real_slices.transpose((1, 0, -1))
cine_dct_gt_slices = cine_dct_gt_slices.transpose((1, 0, -1))
de_dct_gt_slices = de_dct_gt_slices.transpose((1, 0, -1))

# remove slices
cine_real_slices = np.delete(cine_real_slices, cine_target_slices, axis=2)
de_real_slices = np.delete(de_real_slices, de_target_slices, axis=2)
cine_gt_slices = np.delete(cine_gt_slices, cine_gt_target_slices, axis=2)
de_gt_slices = np.delete(de_gt_slices, de_gt_target_slices, axis=2)

#dct
cine_dct_real_slices = np.delete(cine_dct_real_slices, cine_dct_target_slices, axis=2)
de_dct_real_slices = np.delete(de_dct_real_slices, de_dct_target_slices, axis=2)
cine_dct_gt_slices = np.delete(cine_dct_gt_slices, cine_dct_gt_target_slices, axis=2)
de_dct_gt_slices = np.delete(de_dct_gt_slices, de_dct_gt_target_slices, axis=2)


# save the new nifti files
# print ((cine_img.header))
save_nifti(cine_real_slices, cine_final_path, cine_case.split(os.path.sep)[0] + "_frame01",
           constant.IS_NOT_GT, cine_img.header, constant.CINE_TYPE, False)
save_nifti(de_real_slices, de_real_final_path, de_case.split(os.path.sep)[0], constant.IS_NOT_GT, de_img.header,
           constant.DE_TYPE, False)
save_nifti(cine_gt_slices, cine_final_path, cine_gt_case.split(os.path.sep)[0] + "_frame01",
           constant.IS_GT, cine_gt.header, constant.CINE_TYPE, False)
save_nifti(de_gt_slices, de_gt_final_path, "Case_P" + str(de_number), constant.IS_GT, de_gt.header, constant.DE_TYPE, False)

#dct
save_nifti(cine_dct_real_slices, cine_dct_final_path, cine_dct_case.split(os.path.sep)[0] + "_frame01",
           constant.IS_NOT_GT, cine_dct_img.header, constant.CINE_TYPE, True)
save_nifti(de_dct_real_slices, de_dct_real_final_path, de_dct_case.split(os.path.sep)[0], constant.IS_NOT_GT, de_dct_img.header,
           constant.DE_TYPE, True)
save_nifti(cine_dct_gt_slices, cine_dct_final_path, cine_dct_gt_case.split(os.path.sep)[0] + "_frame01",
           constant.IS_GT, cine_dct_gt.header, constant.CINE_TYPE, True)
save_nifti(de_dct_gt_slices, de_dct_gt_final_path, "Case_P" + str(de_number), constant.IS_GT, de_dct_gt.header, constant.DE_TYPE, True)

# Remove the temporary Slices
utils.remove_contents(constant.CINE_SAVE_PATH)
utils.remove_contents(constant.DE_SAVE_PATH)
utils.remove_contents(constant.CINE_MASK_SAVE_PATH)
utils.remove_contents(constant.DE_MASK_SAVE_PATH)

#dct
utils.remove_contents(constant.CINE_DCT_SAVE_PATH)
utils.remove_contents(constant.DE_DCT_SAVE_PATH)
utils.remove_contents(constant.CINE_DCT_MASK_SAVE_PATH)
utils.remove_contents(constant.DE_DCT_MASK_SAVE_PATH)