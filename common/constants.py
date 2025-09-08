import os

# Path of images
# ROOT = "/home/oem/multi-mri-seg_project/multi-mri-seg/data"
ROOT = "E:\internship\code\multiple_MRI\multi-mri-seg\data"
CINE_DATA_PATH = os.path.join(ROOT, 'ACDC/training')
DE_DATA_PATH = os.path.join(ROOT, 'Emidec_Test_Images')
DE_GT_DATA_PATH = os.path.join(ROOT, 'Emidec_Test_GT')

EPI_CINE_DATA_PATH_WHOLE = os.path.join(ROOT, 'EPI_ACDC_whole/training')
EPI_DE_GT_DATA_PATH_WHOLE = os.path.join(ROOT, 'EPI_Emidec_whole')

EPI_CINE_DATA_PATH_ROI = os.path.join(ROOT, 'EPI_ACDC_roi/training')
EPI_DE_GT_DATA_PATH_ROI = os.path.join(ROOT, 'EPI_Emidec_roi')

EPI_CINE_DATA_PATH_ROI_DCT = os.path.join(ROOT, 'EPI_ACDC_roi_dct/training')
EPI_DE_GT_DATA_PATH_ROI_DCT = os.path.join(ROOT, 'EPI_Emidec_roi_dct')

CINEDE_RGH_DATA_PATH = os.path.join(ROOT, 'CINEDE_RGH/training')
CINEDE_RGH_AB_DATA_PATH = os.path.join(ROOT, 'CINEDE_RGH_AB/training')
CINEDE_RGH_BLUR_DATA_PATH = os.path.join(ROOT, 'CINEDE_272_NON_REG/training')

# CINE_SAVE_PATH = "../../core/dataManagement/CINE_MRI"
# DE_SAVE_PATH = "../../core/dataManagement/DE_MRI"
# CINE_MASK_SAVE_PATH = "../../core/dataManagement/CINE_MASK_MRI"
# DE_MASK_SAVE_PATH = "../../core/dataManagement/DE_MASK_MRI"
#
# CINE_DCT_SAVE_PATH = "../../core/dataManagement/CINE_DCT_MRI"
# DE_DCT_SAVE_PATH = "../../core/dataManagement/DE_DCT_MRI"
# CINE_DCT_MASK_SAVE_PATH = "../../core/dataManagement/CINE_DCT_MASK_MRI"
# DE_DCT_MASK_SAVE_PATH = "../../core/dataManagement/DE_DCT_MASK_MRI"
###
CINE_SAVE_PATH = "../core/dataManagement/CINE_MRI"
DE_SAVE_PATH = "../core/dataManagement/DE_MRI"
CINE_MASK_SAVE_PATH = "../core/dataManagement/CINE_MASK_MRI"
DE_MASK_SAVE_PATH = "../core/dataManagement/DE_MASK_MRI"

CINE_DCT_SAVE_PATH = "../core/dataManagement/CINE_DCT_MRI"
DE_DCT_SAVE_PATH = "../core/dataManagement/DE_DCT_MRI"
CINE_DCT_MASK_SAVE_PATH = "../core/dataManagement/CINE_DCT_MASK_MRI"
DE_DCT_MASK_SAVE_PATH = "../core/dataManagement/DE_DCT_MASK_MRI"
####
CINE_ADAPTED_DATA_PATH_GENERAL = os.path.join(ROOT, 'ACDC_adapted_general/training')
DE_ADAPTED_DATA_PATH_GENERAL = os.path.join(ROOT, 'Emidec_adapted_general')

CINE_ADAPTED_DATA_PATH_WHOLE = os.path.join(ROOT, 'ACDC_adapted_whole/training')
DE_ADAPTED_DATA_PATH_WHOLE = os.path.join(ROOT, 'Emidec_adapted_whole')

CINE_ADAPTED_DATA_PATH_ROI = os.path.join(ROOT, 'ACDC_adapted_roi/training')
DE_ADAPTED_DATA_PATH_ROI = os.path.join(ROOT, 'Emidec_adapted_roi')

CINE_ADAPTED_DATA_PATH_ROI_DTC = os.path.join(ROOT, 'ACDC_adapted_roi_dct/training')
DE_ADAPTED_DATA_PATH_ROI_DCT = os.path.join(ROOT, 'Emidec_adapted_roi_dct')

CINE_ADAPTED_DATA_PATH_ROI_EH_DCT = os.path.join(ROOT, 'ACDC_adapted_roi_eh_dct/training')
DE_ADAPTED_DATA_PATH_ROI_EH_DCT = os.path.join(ROOT, 'Emidec_adapted_roi_eh_dct')
# DE_ADAPTED_GT_SAVE_PATH = "../core/dataManagement/DE_MRI_GT_RES_ADAPTED"

DE_RES_ADAPTED_SAVE_PATH = os.path.join(ROOT, 'Emidec_MRI_Res_Adapted')
DE_RES_ADAPTED_GT_SAVE_PATH = os.path.join(ROOT, 'Emidec_MRI_GT_Res_Adapted')
CINE_RES_ADAPTED_SAVE_PATH = os.path.join(ROOT, 'ACDC_MRI_Res_Adapted')
CINE_RES_ADAPTED_GT_SAVE_PATH = os.path.join(ROOT, 'ACDC_MRI_GT_Res_Adapted')
REGISTRATION_DATA_PATH = os.path.join(ROOT, 'Registration')

TEST = "E:\internship\code\multiple_MRI\multi-mri-seg\data\CINDE_RG\\testing"

# Path of the association file
ASSOCIATION_PATH = "../configuration/association.txt"
CINE_TYPE = 0
DE_TYPE = 1

DIMENSION_HEIGHT = 0
DIMENSION_WIDTH = 1

DO_SAVE = True
DO_NOT_SAVE = False

IS_GT = True
IS_NOT_GT = False

IS_REVERSED = True
IS_NOT_REVERSED = False

IS_KIND_OF_OVERLAY = True

# database options
WHOLE_IMAGE = 1
ROI = 2
ROI_OBJ_DETECTION = 3
ROI_ENHANCEMENT_OBJ_DETECTION = 4
NOT_REGISTRED = 5

# loading options
REAL_IMAGE = 1
MASK = 2
OVERLAY = 3
OPPOSITE_OVERLAY = 4

LOGO_PATH = "icons/logo.png"

# Styles
# colorHeader = '#2a9d8f'
colorHeader = '#05668d'
colorHeaderTab = '#005580'
colorContent = "#f2f2f2"
colorSelected = "#005580"
colorUnSelected = "#b1cae7"
# colorButton = "#05668d"
colorButton = "#2a9d8f"
colorExitButton = "#ff0000"
colorButtonText = "#ffffff"
colorActionLabel = "#05668d"
colorActionBg = "#ffffff"
colorSuccessBG = "#def1de"
colorSuccessFG = "#5cb85c"
colorImageNameBG = "#e6f2ff"
colorSlidebar = "#e0ebeb"
colorHausdorffFG = "#ff9966"

FOOT_PAGE_DATA = "Azadeh Hadadi, Final Thesis, MSCV2, ImViA Laboratory, University of Burgundy, February 2021"

