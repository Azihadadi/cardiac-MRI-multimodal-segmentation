import os

import PIL
import nibabel as nib
import numpy as np
import cv2
from PIL import Image

import common.constants as constant
import common.utils as utils
import SimpleITK as sitk
import imutils

from keras.preprocessing.image import img_to_array
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from skimage.metrics import structural_similarity as ssim


# define the test configuration
class TestConfig(Config):
    NAME = "ventricle"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 5


class DataManager:
    # load slices
    def load_slices(self, data, path, do_save):
        # transpose to convert x,y to height,width
        data = data.transpose((1, 0, -1))
        for slice_number in range(data.shape[2]):
            real_img = data[:, :, slice_number]

            img_fullname = path + "/slice_" + str(slice_number) + ".png"
            # Normalization
            normalized_img = np.zeros((data.shape[0], data.shape[1]))
            normalized_img = cv2.normalize(real_img, normalized_img, 0, 255, cv2.NORM_MINMAX)

            # Make a directory to save the slices if save mode is TRUE
            if do_save and not os.path.exists(path):
                os.makedirs(path)
            if do_save:
                cv2.imwrite(img_fullname, normalized_img)
        return real_img.shape

    # calculate the adjusting ratio and return new shape
    def calc_ratio(self, width, height, big_voxel_space, small_voxel_space):
        # Read Voxel_Space
        x_big_voxel_space = round(big_voxel_space[1], 5)
        y_big_voxel_space = round(big_voxel_space[2], 5)
        x_small_voxel_space = round(small_voxel_space[1], 5)
        y_small_voxel_space = round(small_voxel_space[2], 5)

        print('MRI with Bigger Voxel Space of is : {},{}'.format(str(x_big_voxel_space),
                                                                 str(y_big_voxel_space)))
        print('MRI with Smaller Voxel Space of is : {},{}'.format(str(x_small_voxel_space),
                                                                  str(y_small_voxel_space)))

        x_ratio = round((x_big_voxel_space / x_small_voxel_space), 5)
        print('x_ratio: {} '.format(str(x_ratio)))
        y_ratio = round((y_big_voxel_space / y_small_voxel_space), 5)
        print('y_ratio: {} '.format(str(y_ratio)))
        return int(width * x_ratio), int(height * y_ratio)

    # adjust the resolution on MRI with bigger voxel_space
    def do_adjust_resolution(self, image_shape, cine_voxel_space, de_voxel_space, path):
        (new_width, new_height) = self.calc_ratio(image_shape[0], image_shape[1], cine_voxel_space, de_voxel_space)

        all_slices = np.ones((new_width, new_height, image_shape[2]), dtype=np.uint8)
        for slice_number in range(image_shape[2]):
            img = Image.open(path + "/slice_" + str(slice_number) + ".png")
            im_resized = img.resize((new_width, new_height), resample=PIL.Image.NEAREST)

            im_resized_rev = np.asarray(im_resized).transpose((1, 0))
            all_slices[:, :, slice_number] = im_resized_rev
        return all_slices

    # zero padding on images to make them the same size
    def do_padding(self, all_resized_slices, ref_image, opposite_path):

        # get the max height and width of slices
        all_resized_slices = all_resized_slices.transpose(1, 0, -1)
        ref_image = ref_image.transpose(1, 0, -1)

        max_height = (max(all_resized_slices.shape[0], ref_image.shape[0]))
        max_width = (max(all_resized_slices.shape[1], ref_image.shape[1]))

        all_slices = np.ones((max_height, max_width, all_resized_slices.shape[2]), dtype=np.uint8)
        all_ref_slices = np.ones((max_height, max_width, ref_image.shape[2]), dtype=np.uint8)

        # the slices which adjusted resolution
        for slice_number in range(all_resized_slices.shape[2]):
            img = all_resized_slices[:, :, slice_number]
            img_padding = np.zeros((max_height, max_width), dtype=np.uint8)
            if img.shape[0] == max_height and img.shape[1] == max_width:
                img_padding = img
            elif img.shape[0] < max_height and img.shape[1] < max_width:
                img_padding = utils.zero_padding(np.asarray(img), constant.DIMENSION_HEIGHT, ref_image.shape[0])
                img_padding = utils.zero_padding(img_padding, constant.DIMENSION_WIDTH, ref_image.shape[1])
            elif img.shape[0] < max_height:
                img_padding = utils.zero_padding(np.asarray(img), constant.DIMENSION_HEIGHT, ref_image.shape[0])
            elif img.shape[1] < max_width:
                img_padding = utils.zero_padding(np.asarray(img), constant.DIMENSION_WIDTH, ref_image.shape[1])

            all_slices[:, :, slice_number] = img_padding

        # reference slices which not adjusted resolution
        for slice_number in range(ref_image.shape[2]):
            img = Image.open(opposite_path + "/slice_" + str(slice_number) + ".png")
            img_padding = np.zeros((max_height, max_width), dtype=np.uint8)
            if ref_image.shape[0] == max_height and ref_image.shape[1] == max_width:
                img_padding = ref_image[:, :, slice_number]
            elif ref_image.shape[0] < max_height and ref_image.shape[1] < max_width:
                img_padding = utils.zero_padding(np.asarray(img), constant.DIMENSION_HEIGHT,
                                                 all_resized_slices.shape[0])
                img_padding = utils.zero_padding(img_padding, constant.DIMENSION_WIDTH, all_resized_slices.shape[1])
            elif ref_image.shape[0] < max_height:
                img_padding = utils.zero_padding(np.asarray(img), constant.DIMENSION_HEIGHT,
                                                 all_resized_slices.shape[0])
            elif ref_image.shape[1] < max_width:
                img_padding = utils.zero_padding(np.asarray(img), constant.DIMENSION_WIDTH, all_resized_slices.shape[1])
            all_ref_slices[:, :, slice_number] = img_padding

        return all_slices, all_ref_slices

    def command_iteration(self, method):
        print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")

    def evaluation_single_image(self, img, moving_roi, y_start_cine, x_start_cine, margine):
        y_start_de = 0
        x_start_de = 0
        height_de = 0
        width_de = 0
        orig_img = img.copy()

        # define the model
        rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())

        # load model weights
        model_path = ("../core/dataManagement/models/1073_epi_0011.h5")
        rcnn.load_weights(model_path, by_name=True)

        # load photograph
        img = img_to_array(img)

        # make prediction
        results = rcnn.detect([img], verbose=0)
        # get dictionary for first prediction
        r = results[0]
        all_detected_boxes = r['rois']
        valid_box_ssim = []
        if len(all_detected_boxes) == 0:
            return (y_start_de, x_start_de, height_de, width_de)
        if len(all_detected_boxes) > 1:
            for i in range(len(all_detected_boxes)):
                y1, x1, y2, x2 = all_detected_boxes[i]
                if y1 > y_start_cine - margine and y1 < y_start_cine + margine and x1 > x_start_cine - margine and x1 < x_start_cine + margine:
                    region = orig_img[y1:y2, x1:x2]
                    mean_intensity = np.mean(region)
                    print("Mean Pixel Values: " + str(mean_intensity))
                    print(all_detected_boxes)
                    (res_cine_roi, res_region) = utils.make_same_size(moving_roi, region)
                    s = ssim(res_cine_roi, res_region)
                    print("******** S:" + str(s))
                    if s > 0.1:
                        # if mean_intensity > 120:
                        valid_box_ssim.append((all_detected_boxes[i], s))
                        # new_boxes.append(boxes[i])
        else:
            valid_box_ssim.append((all_detected_boxes[0], 1))

        if len(valid_box_ssim) > 1:
            valid_box_ssim.sort(key=utils.take_element, reverse=True)

        boxes = utils.take_list(valid_box_ssim)
        for i in range(len(boxes)):
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            y_start_de = y1
            x_start_de = x1
            height_de = y2 - y1
            width_de = x2 - x1
            if y_start_de > y_start_cine - margine and y_start_de < y_start_cine + margine and x_start_de > x_start_cine - margine and x_start_de < x_start_cine + margine:
                return (y_start_de, x_start_de, height_de, width_de)
        return (0, 0, 0, 0)

    def do_registration_roi_dct(self, cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices):
        # the slices which adjusted resolution
        x_margine = y_margine = 10
        for slice_number in range(cine_gt_slices.shape[2]):
            print("******************* Slice Number: " + str(slice_number) + "********************")
            fixed_gt = de_gt_slices[:, :, slice_number]
            fixed_real = de_real_slices[:, :, slice_number]
            moving_gt = cine_gt_slices[:, :, slice_number]
            moving_real = cine_real_slices[:, :, slice_number]

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            fixed_real = clahe.apply(fixed_real)

            # find external contours in the fixed_gt (CINE mask)
            cnts = cv2.findContours(moving_gt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # fit a bounding box to the contour
            (x, y, w, h) = cv2.boundingRect(cnts[0])

            fixed_copy = np.zeros(fixed_real.shape, dtype=np.uint8)
            moving_copy = np.zeros(moving_real.shape, dtype=np.uint8)

            # template = cv2.equalizeHist(template)
            (y_start_de, x_start_de, height_de, width_de) = self.evaluation_single_image(fixed_real,
                                                                                         moving_real[y:y + h, x:x + w],
                                                                                         y, x, 50)
            # extract the region from moving real image (DE image)
            if height_de == width_de == 0:
                # de_roi = fixed_real[y - y_margine:y + y_margine + h, x - x_margine:x + x_margine + w]
                # cine_roi = moving_real[y - y_margine:y + y_margine + h, x - x_margine:x + x_margine + w]
                # fixed_copy[y - y_margine:y + y_margine + h, x - x_margine:x + x_margine + w] = de_roi
                # moving_copy[y - y_margine:y + y_margine + h, x - x_margine:x + x_margine + w] = cine_roi
                fixed_copy = fixed_real
                moving_copy = moving_real
            else:
                de_roi = fixed_real[y_start_de:y_start_de + height_de, x_start_de:x_start_de + width_de]
                cine_roi = moving_real[y:y + h, x:x + w]
                fixed_copy[y_start_de:y_start_de + height_de, x_start_de:x_start_de + width_de] = de_roi
                moving_copy[y:y + h, x:x + w] = cine_roi
            # moving_app_region = cv2.equalizeHist(moving_app_region)

            fixed_ROI_sitk = sitk.Cast(sitk.GetImageFromArray(np.asarray(fixed_copy)), sitk.sitkFloat32)
            moving_ROI_sitk = sitk.Cast(sitk.GetImageFromArray(np.asarray(moving_copy)), sitk.sitkFloat32)

            fixed_gt_sitk = sitk.Cast(sitk.GetImageFromArray(fixed_gt),
                                      sitk.sitkFloat32)
            fixed_real_sitk = sitk.Cast(sitk.GetImageFromArray(fixed_real),
                                        sitk.sitkFloat32)
            moving_gt_sitk = sitk.Cast(sitk.GetImageFromArray(moving_gt),
                                       sitk.sitkFloat32)
            moving_real_sitk = sitk.Cast(sitk.GetImageFromArray(moving_real),
                                         sitk.sitkFloat32)

            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMeanSquares()
            R.SetOptimizerAsRegularStepGradientDescent(4.0, .0001, 200)
            R.SetInitialTransform(sitk.TranslationTransform(fixed_ROI_sitk.GetDimension()))
            R.SetInterpolator(sitk.sitkLinear)
            R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(R))
            outTx = R.Execute(fixed_ROI_sitk, moving_ROI_sitk)
            print("-------")
            print(outTx)
            print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
            print(f" Iteration: {R.GetOptimizerIteration()}")
            print(f" Metric value: {R.GetMetricValue()}")

            resampler_real = sitk.ResampleImageFilter()
            resampler_real.SetReferenceImage(fixed_real_sitk)
            resampler_real.SetInterpolator(sitk.sitkLinear)
            resampler_real.SetDefaultPixelValue(0)
            resampler_real.SetTransform(outTx)

            resampler_gt = sitk.ResampleImageFilter()
            resampler_gt.SetReferenceImage(fixed_gt_sitk)
            resampler_gt.SetInterpolator(sitk.sitkLinear)
            resampler_gt.SetDefaultPixelValue(0)
            resampler_gt.SetTransform(outTx)

            out_real = resampler_real.Execute(moving_real_sitk)
            out_gt = resampler_gt.Execute(moving_gt_sitk)

            simg_real = sitk.Cast(sitk.RescaleIntensity(out_real), sitk.sitkUInt8)
            simg_gt = sitk.Cast(sitk.RescaleIntensity(out_gt), sitk.sitkUInt8)

            cine_gt_slices[:, :, slice_number] = sitk.GetArrayFromImage(simg_gt)
            cine_real_slices[:, :, slice_number] = sitk.GetArrayFromImage(simg_real)
            de_real_slices[:, :, slice_number] = fixed_real
        # return the all of the slices
        return cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices

    def do_registration_roi_enh_dct(self, cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices):
        # the slices which adjusted resolution
        for slice_number in range(cine_gt_slices.shape[2]):
            fixed_gt = de_gt_slices[:, :, slice_number]
            fixed_real = de_real_slices[:, :, slice_number]
            moving_gt = cine_gt_slices[:, :, slice_number]
            moving_real = cine_real_slices[:, :, slice_number]

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            fixed_real = clahe.apply(fixed_real)

            fixed_gt = sitk.Cast(sitk.GetImageFromArray(np.asarray(fixed_gt)),
                                 sitk.sitkFloat64)
            fixed_real = sitk.Cast(sitk.GetImageFromArray(np.asarray(fixed_real)),
                                   sitk.sitkFloat64)
            moving_gt = sitk.Cast(sitk.GetImageFromArray(np.asarray(moving_gt)),
                                  sitk.sitkFloat64)
            moving_real = sitk.Cast(sitk.GetImageFromArray(np.asarray(moving_real)),
                                    sitk.sitkFloat64)

            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMeanSquares()
            R.SetOptimizerAsRegularStepGradientDescent(4.0, .0001, 200)
            R.SetInitialTransform(sitk.TranslationTransform(fixed_gt.GetDimension()))
            R.SetInterpolator(sitk.sitkLinear)
            R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(R))

            outTx = R.Execute(fixed_gt, moving_gt)
            print("-------")
            print(outTx)
            print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
            print(f" Iteration: {R.GetOptimizerIteration()}")
            print(f" Metric value: {R.GetMetricValue()}")


            resampler_real = sitk.ResampleImageFilter()
            resampler_real.SetReferenceImage(fixed_gt)
            resampler_real.SetInterpolator(sitk.sitkLinear)
            resampler_real.SetDefaultPixelValue(0)
            resampler_real.SetTransform(outTx)

            resampler_gt = sitk.ResampleImageFilter()
            resampler_gt.SetReferenceImage(fixed_real)
            resampler_gt.SetInterpolator(sitk.sitkLinear)
            resampler_gt.SetDefaultPixelValue(0)
            resampler_gt.SetTransform(outTx)

            out_real = resampler_real.Execute(moving_real)
            out_gt = resampler_gt.Execute(moving_gt)

            simg_real = sitk.Cast(sitk.RescaleIntensity(out_real), sitk.sitkUInt8)
            simg_gt = sitk.Cast(sitk.RescaleIntensity(out_gt), sitk.sitkUInt8)

            cine_gt_slices[:, :, slice_number] = sitk.GetArrayFromImage(simg_gt)
            cine_real_slices[:, :, slice_number] = sitk.GetArrayFromImage(simg_real)
            # de_real_slices[:, :, slice_number] = sitk.GetArrayFromImage(fixed_real)
        # return the all of the slices
        return cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices

    def do_registration_whole(self, cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices):
        # the slices which adjusted resolution
        for slice_number in range(cine_gt_slices.shape[2]):
            fixed_gt = de_gt_slices[:, :, slice_number]
            fixed_real = de_real_slices[:, :, slice_number]
            moving_gt = cine_gt_slices[:, :, slice_number]
            moving_real = cine_real_slices[:, :, slice_number]

            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # fixed_real = clahe.apply(fixed_real)

            # fixed_real = img_as_float(fixed_real)
            # fixed_real = bm3d.bm3d(fixed_real, sigma_psd=0.5, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            # fixed_real = cv2.fastNlMeansDenoising(fixed_real, 10, 7, 21)

            fixed_gt = sitk.Cast(sitk.GetImageFromArray(np.asarray(fixed_gt)),
                                 sitk.sitkFloat32)
            fixed_real = sitk.Cast(sitk.GetImageFromArray(np.asarray(fixed_real)),
                                   sitk.sitkFloat32)
            moving_gt = sitk.Cast(sitk.GetImageFromArray(np.asarray(moving_gt)),
                                  sitk.sitkFloat32)
            moving_real = sitk.Cast(sitk.GetImageFromArray(np.asarray(moving_real)),
                                    sitk.sitkFloat32)

            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMeanSquares()
            R.SetOptimizerAsRegularStepGradientDescent(4.0, .0001, 200)
            R.SetInitialTransform(sitk.TranslationTransform(fixed_real.GetDimension()))
            R.SetInterpolator(sitk.sitkLinear)
            R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(R))

            outTx = R.Execute(fixed_real, moving_real)
            print("-------")
            print(outTx)
            print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
            print(f" Iteration: {R.GetOptimizerIteration()}")
            print(f" Metric value: {R.GetMetricValue()}")

            if ("SITK_NOSHOW" not in os.environ):
                resampler_real = sitk.ResampleImageFilter()
                resampler_real.SetReferenceImage(fixed_real)
                resampler_real.SetInterpolator(sitk.sitkLinear)
                resampler_real.SetDefaultPixelValue(0)
                resampler_real.SetTransform(outTx)

            resampler_gt = sitk.ResampleImageFilter()
            resampler_gt.SetReferenceImage(fixed_gt)
            resampler_gt.SetInterpolator(sitk.sitkLinear)
            resampler_gt.SetDefaultPixelValue(0)
            resampler_gt.SetTransform(outTx)

            out_real = resampler_real.Execute(moving_real)
            out_gt = resampler_gt.Execute(moving_gt)

            simg_real = sitk.Cast(sitk.RescaleIntensity(out_real), sitk.sitkUInt8)
            simg_gt = sitk.Cast(sitk.RescaleIntensity(out_gt), sitk.sitkUInt8)

            cine_gt_slices[:, :, slice_number] = sitk.GetArrayFromImage(simg_gt)
            cine_real_slices[:, :, slice_number] = sitk.GetArrayFromImage(simg_real)
            # de_real_slices[:, :, slice_number] = sitk.GetArrayFromImage(fixed_real)
        # return the all of the slices
        return cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices

    def do_registration_roi(self, cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices):
        # the slices which adjusted resolution
        x_margine = y_margine = 20
        for slice_number in range(cine_gt_slices.shape[2]):
            fixed_gt = de_gt_slices[:, :, slice_number]
            fixed_real = de_real_slices[:, :, slice_number]
            moving_gt = cine_gt_slices[:, :, slice_number]
            moving_real = cine_real_slices[:, :, slice_number]

            # adaptive histogram equalization
            # create a CLAHE object (Arguments are optional).
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # fixed_real = clahe.apply(fixed_real)


            # find external contours in the moving_gt (CINE mask)
            cnts = cv2.findContours(moving_gt.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # fit a bounding box to the contour
            (x, y, w, h) = cv2.boundingRect(cnts[0])

            fixed_copy = np.zeros(fixed_real.shape, dtype=np.uint8)
            moving_copy = np.zeros(moving_real.shape, dtype=np.uint8)

            de_roi = fixed_real[y - y_margine:y + y_margine + h, x - x_margine:x + x_margine + w]
            cine_roi = moving_real[y - y_margine:y + y_margine + h, x - x_margine:x + x_margine + w]

            # fixed_copy[y:y + h, x:x + w] = cine_roi
            fixed_copy[y - y_margine:y + y_margine + h, x - x_margine:x + x_margine + w] = de_roi
            moving_copy[y - y_margine:y + y_margine + h, x - x_margine:x + x_margine + w] = cine_roi

            fixed_ROI_sitk = sitk.Cast(sitk.GetImageFromArray(np.asarray(fixed_copy)), sitk.sitkFloat32)
            moving_ROI_sitk = sitk.Cast(sitk.GetImageFromArray(np.asarray(moving_copy)), sitk.sitkFloat32)

            fixed_gt_sitk = sitk.Cast(sitk.GetImageFromArray(fixed_gt),
                                      sitk.sitkFloat32)
            fixed_real_sitk = sitk.Cast(sitk.GetImageFromArray(fixed_real),
                                        sitk.sitkFloat32)
            moving_gt_sitk = sitk.Cast(sitk.GetImageFromArray(moving_gt),
                                       sitk.sitkFloat32)
            moving_real_sitk = sitk.Cast(sitk.GetImageFromArray(moving_real),
                                         sitk.sitkFloat32)

            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMeanSquares()
            R.SetOptimizerAsRegularStepGradientDescent(4.0, .0001, 200)
            R.SetInitialTransform(sitk.TranslationTransform(fixed_ROI_sitk.GetDimension()))
            R.SetInterpolator(sitk.sitkLinear)
            R.AddCommand(sitk.sitkIterationEvent, lambda: self.command_iteration(R))

            outTx = R.Execute(fixed_ROI_sitk, moving_ROI_sitk)
            print("-------")
            print(outTx)
            print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
            print(f" Iteration: {R.GetOptimizerIteration()}")
            print(f" Metric value: {R.GetMetricValue()}")

            resampler_real = sitk.ResampleImageFilter()
            resampler_real.SetReferenceImage(fixed_real_sitk)
            resampler_real.SetInterpolator(sitk.sitkLinear)
            resampler_real.SetDefaultPixelValue(0)
            resampler_real.SetTransform(outTx)

            resampler_gt = sitk.ResampleImageFilter()
            resampler_gt.SetReferenceImage(fixed_gt_sitk)
            resampler_gt.SetInterpolator(sitk.sitkLinear)
            resampler_gt.SetDefaultPixelValue(0)
            resampler_gt.SetTransform(outTx)

            out_real = resampler_real.Execute(moving_real_sitk)
            out_gt = resampler_gt.Execute(moving_gt_sitk)

            simg_real = sitk.Cast(sitk.RescaleIntensity(out_real), sitk.sitkUInt8)
            simg_gt = sitk.Cast(sitk.RescaleIntensity(out_gt), sitk.sitkUInt8)

            cine_gt_slices[:, :, slice_number] = sitk.GetArrayFromImage(simg_gt)
            cine_real_slices[:, :, slice_number] = sitk.GetArrayFromImage(simg_real)
            # de_real_slices[:, :, slice_number] = sitk.GetArrayFromImage(fixed_real)
        # return the all of the slices
        return cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices

    # save slices as a new nifti file with file_name in the path
    def save_nifti(self, all_slices, path, file_name, header, is_gt):
        all_slices = np.asarray(all_slices)
        # save as nifti
        if not os.path.exists(path):
            os.makedirs(path)
        file_suffix = '_adapted_gt.nii.gz' if is_gt else "_adapted.nii.gz"
        # file_suffix = '_gt.nii.gz' if is_gt else ".nii.gz"
        if is_gt:
            for slice_number in range(all_slices.shape[2]):
                # normalization (convert 0-->0, 128--> 1, 255-->2)
                normalized_img = np.zeros((all_slices.shape[0], all_slices.shape[1]))
                normalized_img = cv2.normalize(all_slices[:, :, slice_number], normalized_img, 0, 2, cv2.NORM_MINMAX)
                all_slices[:, :, slice_number] = normalized_img

        new_ref_nifti = nib.Nifti1Image(all_slices, np.eye(4),header=header.copy())
        nib.save(new_ref_nifti, os.path.join(path, file_name + file_suffix))

        # Remove the temporary Slices
        # utils.remove_contents(path)
        # utils.remove_contents(opposite_path)

    # preprocess flow including: loading niftis, adjusting resolution, zero-padding, registration and save as new nifti files
    def preprocess(self, registration_method):
        # read the associations between Cine and DE MRI
        associations = utils.read_association()
        for asc in associations:
            # real image processing
            cine_case = os.path.join("patient" + str(asc[0]), "patient" + str(asc[0]) + "_frame01.nii.gz")
            print(cine_case)
            de_case = os.path.join("Case_" + str(asc[1]), "Images", "Case_" + str(asc[1]) + ".nii.gz")

            # CINE_MRI
            cine_img = nib.load(os.path.join(constant.CINE_DATA_PATH, cine_case))
            cine_data = cine_img.get_fdata()

            # DE_MRI
            de_img = nib.load(os.path.join(constant.DE_DATA_PATH, de_case))
            de_data = de_img.get_fdata()

            # load the slices
            self.load_slices(cine_data, constant.CINE_SAVE_PATH, constant.DO_SAVE)
            self.load_slices(de_data, constant.DE_SAVE_PATH, constant.DO_SAVE)

            # find the image with bigger voxe_space and then adjusting its resolution, zero_padding, removing the
            # extra slices and saving as a new nifti
            if cine_img.header['pixdim'][1] > de_img.header['pixdim'][1]:
                res = self.do_adjust_resolution((cine_data.shape[0], cine_data.shape[1], cine_data.shape[2]),
                                                cine_img.header['pixdim'], de_img.header['pixdim'],
                                                constant.CINE_SAVE_PATH)
                (cine_real_slices, de_real_slices) = self.do_padding(res, de_data, constant.DE_SAVE_PATH)
            else:
                res = self.do_adjust_resolution((de_data.shape[0], de_data.shape[1], de_data.shape[2]),
                                                de_img.header['pixdim'], cine_img.header['pixdim'],
                                                constant.DE_SAVE_PATH)
                (de_real_slices, cine_real_slices) = self.do_padding(res, cine_data, constant.CINE_SAVE_PATH)

            # ground truth processing
            cine_gt_case = os.path.join("patient" + str(asc[0]), "patient" + str(asc[0]) + "_frame01_gt.nii.gz")
            de_gt_case = os.path.join("Case_" + str(asc[1]), "Contours", "Case_P" + str(asc[1]) + ".nii.gz")

            # CINE_MRI GT
            cine_gt = nib.load(os.path.join(constant.CINE_DATA_PATH, cine_gt_case))
            cine_gt_data = cine_gt.get_fdata()

            # DE_MRI GT
            de_gt = nib.load(os.path.join(constant.DE_GT_DATA_PATH, de_gt_case))
            de_gt_data = de_gt.get_fdata()

            # load the slices
            self.load_slices(cine_gt_data, constant.CINE_MASK_SAVE_PATH,
                             constant.DO_SAVE)
            self.load_slices(de_gt_data, constant.DE_MASK_SAVE_PATH,
                             constant.DO_SAVE)

            # find the image with bigger voxe_space and then adjust its resolution, zero_padding and save as a new nifti
            if cine_img.header['pixdim'][1] > de_img.header['pixdim'][1]:
                res_gt = self.do_adjust_resolution(
                    (cine_gt_data.shape[0], cine_gt_data.shape[1], cine_gt_data.shape[2]), cine_gt.header['pixdim'],
                    de_gt.header['pixdim'],
                    constant.CINE_MASK_SAVE_PATH)
                (cine_gt_slices, de_gt_slices) = self.do_padding(res_gt, de_gt_data, constant.DE_MASK_SAVE_PATH)
            else:
                res_gt = self.do_adjust_resolution((de_gt_data.shape[0], de_gt_data.shape[1], de_gt_data.shape[2]),
                                                   de_gt.header['pixdim'], cine_gt.header['pixdim'],
                                                   constant.DE_MASK_SAVE_PATH)
                (de_gt_slices, cine_gt_slices) = self.do_padding(res_gt, cine_gt_data, constant.CINE_MASK_SAVE_PATH)

            if registration_method.get() == constant.WHOLE_IMAGE:
                (cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices) = self.do_registration_whole(
                    cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices)
                base_cine_adapted_path = constant.CINE_ADAPTED_DATA_PATH_WHOLE
                base_de_adapted_path = constant.DE_ADAPTED_DATA_PATH_WHOLE
            elif registration_method.get() == constant.ROI:
                (cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices) = self.do_registration_roi(
                    cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices)
                base_cine_adapted_path = constant.CINE_ADAPTED_DATA_PATH_ROI
                base_de_adapted_path = constant.DE_ADAPTED_DATA_PATH_ROI
            elif registration_method.get() == constant.ROI_OBJ_DETECTION:
                (cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices) = self.do_registration_roi_dct(
                    cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices)
                base_cine_adapted_path = constant.CINE_ADAPTED_DATA_PATH_ROI_DTC
                base_de_adapted_path = constant.DE_ADAPTED_DATA_PATH_ROI_DCT
            elif registration_method.get() == constant.ROI_ENHANCEMENT_OBJ_DETECTION:
                (cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices) = self.do_registration_roi_enh_dct(
                    cine_real_slices, de_real_slices, cine_gt_slices, de_gt_slices)
                base_cine_adapted_path = constant.CINE_ADAPTED_DATA_PATH_ROI_EH_DCT
                base_de_adapted_path = constant.DE_ADAPTED_DATA_PATH_ROI_EH_DCT
            elif registration_method.get() == constant.NOT_REGISTRED:
                base_cine_adapted_path = constant.CINE_ADAPTED_DATA_PATH_GENERAL
                base_de_adapted_path = constant.DE_ADAPTED_DATA_PATH_GENERAL

            # define the final path to save the new nifti files
            cine_final_path = os.path.join(base_cine_adapted_path, "patient" + str(asc[0]))
            de_real_final_path = os.path.join(base_de_adapted_path, "Case_" + str(asc[1]), "Images")
            de_gt_final_path = os.path.join(base_de_adapted_path, "Case_" + str(asc[1]), "Contours")
            registration_final_path = os.path.join(constant.REGISTRATION_DATA_PATH,
                                                   "data_" + str(asc[0]) + "_" + str(asc[1]))

            cine_real_slices = cine_real_slices.transpose((1, 0, -1))
            de_real_slices = de_real_slices.transpose((1, 0, -1))
            cine_gt_slices = cine_gt_slices.transpose((1, 0, -1))
            de_gt_slices = de_gt_slices.transpose((1, 0, -1))

            # # save the new nifti files
            self.save_nifti(cine_real_slices, cine_final_path, cine_case.split(os.path.sep)[0] + "_frame01",
                            cine_img.header, constant.IS_NOT_GT)
            self.save_nifti(de_real_slices, de_real_final_path, de_case.split(os.path.sep)[0], de_img.header,
                            constant.IS_NOT_GT)
            self.save_nifti(cine_gt_slices, cine_final_path, cine_gt_case.split(os.path.sep)[0] + "_frame01",
                            cine_gt.header, constant.IS_GT)
            self.save_nifti(de_gt_slices, de_gt_final_path, "Case_P" + str(asc[1]), de_img.header, constant.IS_GT)
            #################################################################################
            # save the new nifti files
            # sorted_files = sorted(os.listdir(os.path.join(constant.ROOT, 'CINEDE_RGH/training')))
            # last_file_name = (sorted_files[-1])
            # last_number = last_file_name[7:].lstrip('0')
            # last_number = int(last_number) + 1
            # main_path = os.path.join(constant.ROOT, 'CINEDE_RGH','training',"patient" + str(str(last_number).zfill(3)))
            #
            # self.save_nifti(cine_real_slices, main_path, "patient" + str(str(last_number).zfill(3)),
            #                 cine_img.header, constant.IS_NOT_GT)
            # self.save_nifti(de_real_slices, main_path, "Case_" + str(str(last_number).zfill(3)), de_img.header,
            #                 constant.IS_NOT_GT)
            # self.save_nifti(cine_gt_slices, main_path, "patient" + str(str(last_number).zfill(3)),
            #                 cine_gt.header, constant.IS_GT)
            # self.save_nifti(de_gt_slices, main_path, "Case_" + str(str(last_number).zfill(3)), de_img.header, constant.IS_GT)
        # total number of corresponding data which are defined in association.txt
        return len(associations)
