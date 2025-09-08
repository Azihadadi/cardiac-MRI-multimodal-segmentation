import torch
import torch.utils.data as data
import nibabel as nib
import glob
import os
import numpy as np
from PIL import Image
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
from learning.datasets import constants
import cv2

"""
    Customized data loader for a segmentation (ACDC) dataset
"""


class DataSetACDC(data.Dataset):

    def __init__(self, folder_path, set, patient_ids, im_size=None, transform=None):
        super(DataSetACDC, self).__init__()
        if im_size is None:
            im_size = [256, 256]
        self.set = set
        self.shape = im_size
        if self.set == 'train' or self.set == 'val':
            subfolder = 'training'
        else:
            subfolder = 'testing'

        self.transform = transform  # could add pre-processing (intensity normalization/clipping, ...), data augmentation...
        self.patient_ids = patient_ids
        self.mask_files, self.img_files = [], []
        self.nifti_dim = []
        self.seg_nifti_headers = []

        for pid in self.patient_ids[0]: # validation part (cross-validation)
            # import es and ed
            base_path = subfolder + '/patient' + "{:03d}".format(pid) + '/patient' + "{:03d}".format(pid) + '_frame*'
            patient_data_path = os.path.join(folder_path, base_path)
            files = sorted(glob.glob(patient_data_path))

            im_ed, seg_ed, im_es, seg_es = load_im_from_nifti(files)  # to match with ITK-Snap
            im_ed_header, seg_ed_header, im_es_header, seg_es_header = load_header_from_nifti(files)

            self.nifti_dim.append(im_ed.shape)  # save dimension for compute hausdorff and saving the prediction as a NIfTI
            self.seg_nifti_headers.append(seg_ed_header) # save headers for saving the prediction as a NIfTI

            for s in range(0, im_ed.shape[2]):
                self.img_files.extend(
                    (self.resize(denoise(im_ed[:, :, s]), 2), self.resize(denoise(im_es[:, :, s]), 2)))
                self.mask_files.extend((self.resize(seg_ed[:, :, s], 0), self.resize(seg_es[:, :, s], 0)))

        if self.patient_ids[1] is not None:
            for pid in self.patient_ids[1]: # training: for cross_validation
                # import es and ed
                base_path = subfolder + '/patient' + "{:03d}".format(pid) + '/patient' + "{:03d}".format(
                    pid) + '_frame*'
                patient_data_path = os.path.join(folder_path, base_path)
                files = sorted(glob.glob(patient_data_path))
                im_ed, seg_ed, im_es, seg_es = load_im_from_nifti(files)
                for s in range(0, im_ed.shape[2]):
                    self.img_files.extend(
                        (self.resize(im_ed[:, :, s], 2), self.resize(denoise(im_es[:, :, s]), 2)))
                    self.mask_files.extend((self.resize(seg_ed[:, :, s], 0), self.resize(seg_es[:, :, s], 0)))

    def __getitem__(self, index):
        img = self.img_files[index]
        img = img[None, :, :]
        mask = self.mask_files[index]
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

    def resize(self, image, interp_option):
        image_resized = np.array(Image.fromarray(image).resize(self.shape, resample=interp_option))
        return image_resized

    def __len__(self):
        return len(self.img_files)

    def get_nbstruct(self):
        return constants.ACDC_NUM_CLASS

    # def get_nifti_slice_numbers(self):


"""
    Customized data loader for a segmentation (Emidec) dataset
"""


class DataSetEmidec(data.Dataset):
    def __init__(self, folder_path, set, patient_ids, im_size=None, transform=None):
        super(DataSetEmidec, self).__init__()
        if im_size is None:
            im_size = [256, 256]
        self.set = set
        self.shape = im_size
        if self.set == 'train' or self.set == 'val':
            subfolder = 'training'
        else:
            subfolder = 'testing'

        self.transform = transform  # could add pre-processing (intensity normalization/clipping, ...), data augmentation...
        self.patient_ids = patient_ids
        self.mask_files, self.img_files = [], []
        self.nifti_dim = []
        self.seg_nifti_headers = []

        for indx in self.patient_ids[0]:
            # import images and masks
            patient_case = os.listdir(os.path.join(folder_path, subfolder))[indx - 1]

            base_path = subfolder + '/' + patient_case
            image_path = base_path + '/Images/Case*'
            mask_path = base_path + '/Contours/Case*'
            patient_data_path = [os.path.join(folder_path, image_path),os.path.join(folder_path, mask_path)]
            files = [glob.glob(patient_data_path[0])[0],glob.glob(patient_data_path[1])[0]]

            im_header, seg_header = load_header_from_nifti(files)
            im, seg = load_im_from_nifti(files)

            self.nifti_dim.append(im.shape) # save dimension for compute hausdorff and saving the prediction as a NIfTI
            self.seg_nifti_headers.append(seg_header) # save headers for saving the prediction as a NIfTI

            for s in range(0, im.shape[2]):
                self.img_files.append(self.resize(denoise(clahe(im[:, :, s])), 2))
                self.mask_files.append(self.resize(seg[:, :, s], 0))

        if self.patient_ids[1] is not None:
            for indx in self.patient_ids[1]: # training: for cross_validation
                # import images and masks
                patient_case = os.listdir(os.path.join(folder_path, subfolder))[indx - 1]
                base_path = subfolder + '/' + patient_case
                image_path = base_path + '/Images/Case*'
                mask_path = base_path + '/Contours/Case*'
                patient_data_path = [os.path.join(folder_path, image_path), os.path.join(folder_path, mask_path)]
                files = [glob.glob(patient_data_path[0])[0], glob.glob(patient_data_path[1])[0]]
                im, seg = load_im_from_nifti(files)

                for s in range(0, im.shape[2]):
                    self.img_files.append(self.resize(denoise(clahe(im[:, :, s])), 2))
                    self.mask_files.append(self.resize(seg[:, :, s], 0))

    def __getitem__(self, index):
        img = self.img_files[index]
        img = img[None, :, :]
        mask = self.mask_files[index]
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

    def resize(self, image, interp_option):
        image_resized = np.array(Image.fromarray(image).resize(self.shape, resample=interp_option))
        return image_resized

    def __len__(self):
        return len(self.img_files)

    def get_nbstruct(self):
        return constants.EMIDEC_NUM_CLASS


"""
    Customized data loader for a segmentation (CINEDE) dataset
"""


class DataSetCINEDE(data.Dataset):
    def __init__(self, folder_path, set, patient_ids, im_size=None, transform=None):
        super(DataSetCINEDE, self).__init__()
        if im_size is None:
            im_size = [256, 256]
        self.set = set
        self.shape = im_size
        if self.set == 'train' or self.set == 'val':
            subfolder = 'training'
        else:
            subfolder = 'testing'

        self.transform = transform  # could add pre-processing (intensity normalization/clipping, ...), data augmentation...
        self.patient_ids = patient_ids
        self.mask_files, self.img_files = [], []
        self.nifti_dim = []
        self.seg_nifti_headers = []

        for pid in self.patient_ids[0]:  # validation part (cross-validation)
            # import de and cine
            base_path = subfolder + '/patient' + "{:03d}".format(pid)
            patient_data_path = os.path.join(folder_path, base_path)
            file_list = []
            for file in os.listdir(patient_data_path):
                file_list.append(os.path.join(patient_data_path, file))
            files = sorted(file_list)
            im_de_header, seg_de_header, im_cine_header, seg_cine_header = load_header_from_nifti(files)
            im_de, seg_de, im_cine, seg_cine = load_im_from_nifti(files)

            self.nifti_dim.append(im_cine.shape) # save dimension for compute hausdorff and saving the prediction as a NIfTI
            self.seg_nifti_headers.append((seg_cine_header, seg_de_header)) # save headers for saving the prediction as a NIfTI

            for s in range(0, im_cine.shape[2]):
                self.img_files.append(
                    (self.resize(clahe(im_cine[:, :, s]), 2), self.resize(denoise(im_de[:, :, s]), 2)))
                self.mask_files.append((self.resize(seg_cine[:, :, s], 0), self.resize(seg_de[:, :, s], 0)))

        if self.patient_ids[1] is not None:
            for pid in self.patient_ids[1]:  # training part (cross-validation)
                # import de and cine
                base_path = subfolder + '/patient' + "{:03d}".format(pid)
                patient_data_path = os.path.join(folder_path, base_path)
                file_list = []
                for file in os.listdir(patient_data_path):
                    file_list.append(os.path.join(patient_data_path, file))
                files = sorted(file_list)
                im_de, seg_de, im_cine, seg_cine = load_im_from_nifti(files)

                self.nifti_dim.append(im_cine.shape)
                for s in range(0, im_cine.shape[2]):
                    self.img_files.append(
                        (self.resize(clahe(im_cine[:, :, s]), 2), self.resize(denoise(im_de[:, :, s]), 2)))
                    self.mask_files.append((self.resize(seg_cine[:, :, s], 0), self.resize(seg_de[:, :, s], 0)))

        print("Image files length: " + str(len(self.img_files)))
        print("Mask files length: " + str(len(self.mask_files)))

    def __getitem__(self, index):
        img = self.img_files[index]
        img1 = img[0][None, :, :]
        img2 = img[1][None, :, :]

        if self.transform:
            img1 = self.transform(img1).float().numpy().transpose(1, 0, 2)
            img2 = self.transform(img2).float().numpy().transpose(1, 0, 2)

        mask = self.mask_files[index]
        mask1 = mask[0]
        mask2 = mask[1]
        return torch.from_numpy(img1).float(), torch.from_numpy(mask1).long(), torch.from_numpy(
            img2).float(), torch.from_numpy(mask2).long()

    def resize(self, image, interp_option):
        image_resized = np.array(Image.fromarray(image).resize(self.shape, resample=interp_option))
        return image_resized

    def __len__(self):
        return len(self.img_files)

    def get_nbstruct(self):
        return constants.CINEDE_NUM_CLASS


"""
    Customized data loader for a segmentation (DE) dataset
"""


class DataSetDE(data.Dataset):
    def __init__(self, folder_path, set, patient_ids, im_size=None, transform=None):
        super(DataSetDE, self).__init__()
        if im_size is None:
            im_size = [256, 256]
        self.set = set
        self.shape = im_size
        if self.set == 'train' or self.set == 'val':
            subfolder = 'training'
        else:
            subfolder = 'testing'

        self.transform = transform  # could add pre-processing (intensity normalization/clipping, ...), data augmentation...
        self.patient_ids = patient_ids
        self.mask_files, self.img_files = [], []
        self.nifti_dim = []
        self.seg_nifti_headers = []

        for pid in self.patient_ids[0]: # validation part (cross-validation)
            # import de
            base_path = subfolder + '/patient' + "{:03d}".format(pid) + '/Case_' + "{:03d}".format(pid) + '*'
            patient_data_path = os.path.join(folder_path, base_path)
            files = sorted(glob.glob(patient_data_path))
            im_de_header, seg_de_header = load_header_from_nifti(files)
            im_de, seg_de = load_im_from_nifti(files)

            self.nifti_dim.append(
                im_de.shape)  # save dimension for compute hausdorff and saving the prediction as a NIfTI
            self.seg_nifti_headers.append(seg_de_header)  # save headers for saving the prediction as a NIfTI

            for s in range(0, im_de.shape[2]):
                self.img_files.append(self.resize(denoise(clahe(im_de[:, :, s])), 2))
                self.mask_files.append(self.resize(seg_de[:, :, s], 0))

        if self.patient_ids[1] is not None:
            for pid in self.patient_ids[1]: # training part (cross-validation)
                # import de
                base_path = subfolder + '/patient' + "{:03d}".format(pid) + '/Case_' + "{:03d}".format(pid) + '*'
                patient_data_path = os.path.join(folder_path, base_path)
                files = sorted(glob.glob(patient_data_path))

                im_de, seg_de = load_im_from_nifti(files)

                for s in range(0, im_de.shape[2]):
                    self.img_files.append(self.resize(denoise(clahe(im_de[:, :, s])), 2))
                    self.mask_files.append(self.resize(seg_de[:, :, s], 0))

        print("Image files length: " + str(len(self.img_files)))
        print("Mask files length: " + str(len(self.mask_files)))

    def __getitem__(self, index):
        img = self.img_files[index]
        img = img[None, :, :]

        if self.transform:
            img = self.transform(img).float().numpy().transpose(1, 0, 2)

        mask = self.mask_files[index]
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()

    def resize(self, image, interp_option):
        image_resized = np.array(Image.fromarray(image).resize(self.shape, resample=interp_option))
        return image_resized

    def __len__(self):
        return len(self.img_files)

    def get_nbstruct(self):
        return constants.CINEDE_NUM_CLASS


def load_im_from_nifti(files):
    ims = []
    for fpath in files:
        im = nib.load(fpath).get_fdata()
        im = correct_orientation(im)
        ims.append(im)
    return ims


def load_header_from_nifti(files):
    headers = []
    for fpath in files:
        im = nib.load(fpath)
        header = im.header
        headers.append(header)
    return headers


def to_one_hot(mask, nclass):
    mask_one_hot = torch.zeros((nclass, mask.shape[1], mask.shape[2]))
    mask_one_hot = mask_one_hot.scatter(0, mask, 1).float()

    return mask_one_hot


def correct_orientation(img):
    transform = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
    img = nib.orientations.apply_orientation(img, transform)
    img = np.rot90(img, axes=(1, 0))
    img = img[:, :, :].copy()
    if img.shape[0] < img.shape[1]:
        img = np.swapaxes(img, 0, 1)
    return img


def denoise(img):
    img_float = img_as_float(img)
    sigma_est = np.mean(estimate_sigma(img_float, multichannel=False))
    img_normalized = denoise_nl_means(img, h=1. * sigma_est, fast_mode=True, patch_size=5, patch_distance=3,
                                      multichannel=False, preserve_range=True)
    return img_normalized


def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh_img = clahe.apply(img.astype(np.uint8))
    return enh_img
