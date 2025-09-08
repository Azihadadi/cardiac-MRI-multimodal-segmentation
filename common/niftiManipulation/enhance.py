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

    file_suffix = ".nii.gz"

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

ROOT = "E:\internship\code\multiple_MRI\multi-mri-seg\data"
CINEDE_RGH = os.path.join(ROOT, 'CINEDE_RGH/training')

# number of cases
for de_number in range (2,68):
    de_eh_dct_case = os.path.join("patient" + str(de_number).zfill(3),"Case_" + str(de_number).zfill(3) + ".nii.gz")

    # DE_DCT_MRI
    de_dct_img = nib.load(os.path.join(CINEDE_RGH, de_eh_dct_case))
    de_dct_data = de_dct_img.get_fdata()

    (de_dct_H, de_dct_W) = load_slices(de_dct_data, constant.DE_DCT_SAVE_PATH, constant.IS_NOT_GT)

    de_dct_real_final_path = os.path.join(CINEDE_RGH, CINEDE_RGH, "patient" + str(de_number).zfill(3))

    de_dct_real_slices = make_slices_arrays(constant.DE_DCT_SAVE_PATH, de_dct_H, de_dct_W)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for slice_number in range(de_dct_real_slices.shape[2]):
        de_dct_real_slices[:,:,slice_number] = clahe.apply(de_dct_real_slices[:,:,slice_number])

    de_dct_real_slices = de_dct_real_slices.transpose((1, 0, -1))
    # save the new nifti files
    save_nifti(de_dct_real_slices, de_dct_real_final_path, "Case_" + str(de_number).zfill(3), constant.IS_NOT_GT, de_dct_img.header,
               constant.DE_TYPE, True)
    # Remove the temporary Slices
    utils.remove_contents(constant.DE_DCT_SAVE_PATH)
