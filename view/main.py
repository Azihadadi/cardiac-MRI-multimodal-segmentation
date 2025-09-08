from tkinter import *
from tkinter import filedialog, ttk
from tkinter.font import Font, BOLD
from tkinter.ttk import Style, Notebook

import cv2
from PIL import Image, ImageTk
import common.constants as constants
import common.utils as utils
from core.dataManagement import DataManager
import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk


class Root(Tk):
    dataset_options = {constants.WHOLE_IMAGE: "Whole Image",
                       constants.ROI: "ROI",
                       constants.ROI_OBJ_DETECTION: "ROI + Object Detection + Enhancement"
                       }
    loading_options = {constants.REAL_IMAGE: "Real Image",
                       constants.MASK: "Mask",
                       constants.OVERLAY: "Overlay"
                       }

    # initialization method and define the components
    def __init__(self):
        super(Root, self).__init__()
        # initialization
        self.cine_slice_number = 0
        self.de_slice_number = 0
        self.total_slices_number = 0
        self.set_stylies()
        # headers
        self.header_up_var = StringVar()
        self.header_up_var.set("Automatic Segmentation of the Myocardium from Multi-Modal MRI")
        self.header_down_var = StringVar()
        self.header_down_var.set("ImViA Laboratory, University of Burgundy")
        self.header_up = Label(self, textvariable=self.header_up_var, relief=RAISED, font=self.font_style_header_up,
                               bg=constants.colorHeader,
                               fg=constants.colorActionBg,
                               bd="0",
                               pady="10")
        self.header_down = Label(self, textvariable=self.header_down_var, relief=RAISED, font=self.fontStyleHeader_down,
                                 bg=constants.colorHeader,
                                 fg=constants.colorActionBg,
                                 bd="0",
                                 pady="10")

        # icon
        self.img = ImageTk.PhotoImage(Image.open(constants.LOGO_PATH))
        self.imageIcon_label = Label(self, image=self.img)
        self.imageIcon_label.img = self.img

        # tab Page
        self.tabControl = Notebook(self)
        self.tab_data_management = Frame(self.tabControl, width=200, height=300)
        self.tab_application = Frame(self.tabControl, width=200, height=300)
        self.tabControl.add(self.tab_data_management, text="Data Management")

        # message
        self.message_label = Label(self.tab_data_management, text=None, font=self.fontStyleTabHeader_down)

        # info label
        self.preprocess_info_label = ttk.Label(self.tab_data_management,
                                               text="Please click on Process to do preprocess operations on the corresponding MRIs.",
                                               foreground=constants.colorSelected,
                                               font=self.fontStyleContent_labelFrame,
                                               background=constants.colorContent,
                                               justify=LEFT)

        # process button
        self.process_button = Button(self.tab_data_management, text="Process", command=self.process_callBack,
                                     bg=constants.colorButton,
                                     fg=constants.colorButtonText,
                                     width=10, font=self.fontStyleButton_labelFrame, relief=FLAT)

        # frame of Images
        self.first_images_label_frame = LabelFrame(self.tab_data_management, width=330,
                                                   height=520,
                                                   bd=2,
                                                   font=self.fontStyleButton_labelFrame,
                                                   foreground=constants.colorHeaderTab,
                                                   background=constants.colorActionBg)
        self.first_images_label_frame.grid_propagate(0)

        self.second_images_label_frame = LabelFrame(self.tab_data_management, width=330,
                                                    height=520,
                                                    bd=2,
                                                    font=self.fontStyleButton_labelFrame,
                                                    foreground=constants.colorHeaderTab,
                                                    background=constants.colorActionBg)
        self.second_images_label_frame.grid_propagate(0)
        self.third_images_label_frame = LabelFrame(self.tab_data_management, width=330,
                                                   height=520,
                                                   bd=2,
                                                   font=self.fontStyleButton_labelFrame,
                                                   foreground=constants.colorHeaderTab,
                                                   background=constants.colorActionBg)
        self.third_images_label_frame.grid_propagate(0)
        # dimensions and slice number information
        self.cine_name_label = Label(self.first_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                     bg=constants.colorActionBg, fg=constants.colorActionLabel)
        self.cine_dim_label = Label(self.first_images_label_frame, text=None, font=self.fontStyleContent_labelFrame,
                                    bg=constants.colorActionBg, fg=constants.colorActionLabel)
        self.cine_slice_number_label = Label(self.first_images_label_frame, text=None,
                                             font=self.fontStyleContent_labelFrame,
                                             bg=constants.colorActionBg, fg=constants.colorActionLabel)
        self.de_name_label = Label(self.second_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                   bg=constants.colorActionBg, fg=constants.colorActionLabel)
        self.de_dim_label = Label(self.second_images_label_frame, text=None, font=self.fontStyleContent_labelFrame,
                                  bg=constants.colorActionBg, fg=constants.colorActionLabel)
        self.de_slice_number_label = Label(self.second_images_label_frame, text=None,
                                           font=self.fontStyleContent_labelFrame,
                                           bg=constants.colorActionBg, fg=constants.colorActionLabel)
        self.reg_name_label = Label(self.third_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                    bg=constants.colorActionBg, fg=constants.colorExitButton)
        self.reg_dim_label = Label(self.third_images_label_frame, text=None, font=self.fontStyleContent_labelFrame,
                                   bg=constants.colorActionBg, fg=constants.colorActionLabel)
        self.average_dice_label = Label(self.third_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                bg=constants.colorActionBg, fg=constants.colorSuccessFG)
        self.epi_dice_label = Label(self.third_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                bg=constants.colorActionBg, fg=constants.colorSuccessFG)
        self.myo_dice_label = Label(self.third_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                bg=constants.colorActionBg, fg=constants.colorSuccessFG)
        self.average_hausdorff_label = Label(self.third_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                bg=constants.colorActionBg, fg=constants.colorHausdorffFG)
        self.epi_hausdorff_label = Label(self.third_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                bg=constants.colorActionBg, fg=constants.colorHausdorffFG)
        self.myo_hausdorff_label = Label(self.third_images_label_frame, text=None, font=self.fontStyleButton_labelFrame,
                                bg=constants.colorActionBg, fg=constants.colorHausdorffFG)
        self.reg_slice_number_label = Label(self.third_images_label_frame, text=None,
                                            font=self.fontStyleContent_labelFrame,
                                            bg=constants.colorActionBg, fg=constants.colorActionLabel)

        # frame of loading of corresponding data
        self.loading_csp_label_frame = LabelFrame(self.tab_data_management, text="Corresponding Data", width=530,
                                                  height=200,
                                                  bd=2,
                                                  font=self.fontStyleButton_labelFrame,
                                                  foreground=constants.colorHeaderTab,
                                                  background=constants.colorActionBg)
        self.loading_csp_label_frame.grid_propagate(0)

        # corresponding data
        self.registration_label = ttk.Label(self.loading_csp_label_frame,
                                            text="Registration: ",
                                            background=constants.colorActionBg,
                                            foreground=constants.colorSelected, font=self.fontStyleContent_labelFrame,
                                            justify=LEFT)
        self.reg_var = IntVar()
        self.var = IntVar()
        self.not_registred_radio = ttk.Radiobutton(self.loading_csp_label_frame, text="Original",
                                                   style='Wild.TRadiobutton',
                                                   variable=self.reg_var,
                                                   value=constants.NOT_REGISTRED, command=self.registred_db_callBack)
        self.whole_image_radio = ttk.Radiobutton(self.loading_csp_label_frame, text="Whole Image",
                                                 style='Wild.TRadiobutton',
                                                 variable=self.reg_var,
                                                 value=constants.WHOLE_IMAGE, command=self.registred_db_callBack)

        self.ROI_radio = ttk.Radiobutton(self.loading_csp_label_frame, text="ROI",
                                         style='Wild.TRadiobutton',
                                         variable=self.reg_var,
                                         value=constants.ROI, command=self.registred_db_callBack)
        self.ROI_obj_detection_radio = ttk.Radiobutton(self.loading_csp_label_frame, text="Object Detection",
                                                       style='Wild.TRadiobutton',
                                                       variable=self.reg_var,
                                                       value=constants.ROI_OBJ_DETECTION,
                                                       command=self.registred_db_callBack)
        self.ROI_enhancement_obj_detection_radio = ttk.Radiobutton(self.loading_csp_label_frame,
                                                                   text="Others",
                                                                   style='Wild.TRadiobutton',
                                                                   variable=self.reg_var,
                                                                   value=constants.ROI_ENHANCEMENT_OBJ_DETECTION,
                                                                   command=self.registred_db_callBack)

        self.corresponding_label = ttk.Label(self.loading_csp_label_frame,
                                             text="Corresponding Cases: ",
                                             background=constants.colorActionBg,
                                             foreground=constants.colorSelected, font=self.fontStyleContent_labelFrame,
                                             justify=LEFT,
                                             padding=2)
        self.de_label = ttk.Label(self.loading_csp_label_frame,
                                  text="DE MRI: ",
                                  background=constants.colorActionBg,
                                  foreground=constants.colorSelected, font=self.fontStyleContent_labelFrame,
                                  justify=LEFT)
        self.cine_label = ttk.Label(self.loading_csp_label_frame,
                                    text="CINE MRI: ",
                                    background=constants.colorActionBg,
                                    foreground=constants.colorSelected, font=self.fontStyleContent_labelFrame,
                                    justify=LEFT)

        associations = utils.read_association()
        self.cine_cases = [seq[0] for seq in associations]
        self.cine_cases.insert(0, "--Select--")

        self.cine_combo = ttk.Combobox(self.loading_csp_label_frame, values=self.cine_cases)
        self.cine_combo.bind("<<ComboboxSelected>>", self.update_de_combobox)
        self.cine_combo.current(0)

        self.de_cases = [seq[1] for seq in associations]
        self.de_cases.insert(0, "--Select--")

        self.de_combo = ttk.Combobox(self.loading_csp_label_frame, values=self.de_cases)
        self.de_combo.bind("<<ComboboxSelected>>", self.update_cine_combobox)
        self.de_combo.current(0)

        self.loading_options_label = ttk.Label(self.loading_csp_label_frame,
                                               text="Loading Options: ",
                                               background=constants.colorActionBg,
                                               foreground=constants.colorSelected,
                                               font=self.fontStyleContent_labelFrame,
                                               justify=LEFT,
                                               padding=2)

        self.real_image_radio = ttk.Radiobutton(self.loading_csp_label_frame, text="Real Image",
                                                style='Wild.TRadiobutton',
                                                variable=self.var,
                                                value=constants.REAL_IMAGE, command=self.load_real_image_callBack)
        self.mask_radio = ttk.Radiobutton(self.loading_csp_label_frame, text="Mask", style='Wild.TRadiobutton',
                                          variable=self.var,
                                          value=constants.MASK, command=self.load_mask_callBack)
        self.overlay_radio = ttk.Radiobutton(self.loading_csp_label_frame, text="Overlay", style='Wild.TRadiobutton',
                                             variable=self.var,
                                             value=constants.OVERLAY, command=self.load_overlay_callBack)
        self.opposite_overlay_radio = ttk.Radiobutton(self.loading_csp_label_frame, text="Opposite Overlay",
                                                      style='Wild.TRadiobutton',
                                                      variable=self.var,
                                                      value=constants.OPPOSITE_OVERLAY,
                                                      command=self.load_ops_overlay_callBack)
        # frame of loading general data
        self.loading_gnl_label_frame = LabelFrame(self.tab_data_management, text="Any Data", width=530,
                                                  height=90,
                                                  bd=2,
                                                  font=self.fontStyleButton_labelFrame,
                                                  foreground=constants.colorHeaderTab,
                                                  background=constants.colorActionBg)
        self.loading_gnl_label_frame.grid_propagate(0)

        self.background_data_label = Label(self.loading_gnl_label_frame,
                                           text="Background Image:",
                                           background=constants.colorActionBg,
                                           foreground=constants.colorSelected, font=self.fontStyleContent_labelFrame)
        self.background_browse_button = Button(self.loading_gnl_label_frame, text="Browse",
                                               command=self.background_browse_callBack,
                                               bg=constants.colorButton,
                                               fg=constants.colorButtonText,
                                               width=7, font=self.fontStyleButton_labelFrame, relief=FLAT)

        self.foreground_data_label = Label(self.loading_gnl_label_frame,
                                           text="Foreground Image:",
                                           background=constants.colorActionBg,
                                           foreground=constants.colorSelected, font=self.fontStyleContent_labelFrame)
        self.foreground_browse_button = Button(self.loading_gnl_label_frame, text="Browse",
                                               command=self.foreground_browse_callBack,
                                               bg=constants.colorButton,
                                               fg=constants.colorButtonText,
                                               width=7, font=self.fontStyleButton_labelFrame, relief=FLAT)

        # first and second images
        self.first_image_label = Label(self.first_images_label_frame, image=None)
        self.second_image_label = Label(self.second_images_label_frame, image=None)
        self.third_image_label = Label(self.third_images_label_frame, image=None)

        # exit button
        self.exit_button = Button(self.tab_data_management, text="Exit", command=self.exit_callBack,
                                  bg=constants.colorExitButton, fg=constants.colorButtonText,
                                  width=10, font=self.fontStyleButton_labelFrame, relief=FLAT)
        # footpage label
        self.footPage_label = Label(self.tab_data_management,
                                    text=constants.FOOT_PAGE_DATA,
                                    font=self.fontStyleLabelFooter, fg=constants.colorSelected,
                                    bg=constants.colorContent)
        # define the position of components
        self.set_positions()
        self.is_csp_mode = True
        self.overlaid_two_images = False

        # disable corresponding frame
        self.foreground_browse_button['state'] = DISABLED
        self.real_image_radio['state'] = DISABLED
        self.mask_radio['state'] = DISABLED
        self.overlay_radio['state'] = DISABLED
        self.opposite_overlay_radio['state'] = DISABLED

        self.slide_bar_label = Label(self.tab_data_management,
                                     text="Slices:",
                                     background=constants.colorContent,
                                     foreground=constants.colorSelected, font=self.fontStyleContent_labelFrame)
        self.slide_bar = Scale(self.tab_data_management, from_=1, orient=HORIZONTAL, length=200, bd=2,
                               highlightbackground=constants.colorButton, troughcolor=constants.colorSlidebar,
                               command=self.slide, bg=constants.colorContent, font=self.fontStyleContent_labelFrame)
        self.slide_bar_label.place_forget()
        self.slide_bar.place_forget()
        # instantiation of DataManager
        self.dataManager = DataManager()

    # set whole image as the registration dataset
    def registred_db_callBack(self):
        self.reset_message()
        if self.reg_var.get() == constants.WHOLE_IMAGE:
            self.cine_registred_dataset_path = constants.CINE_ADAPTED_DATA_PATH_WHOLE
            self.de_registred_dataset_path = constants.DE_ADAPTED_DATA_PATH_WHOLE
        elif self.reg_var.get() == constants.ROI:
            self.cine_registred_dataset_path = constants.CINE_ADAPTED_DATA_PATH_ROI
            self.de_registred_dataset_path = constants.DE_ADAPTED_DATA_PATH_ROI
        elif self.reg_var.get() == constants.ROI_OBJ_DETECTION:
            self.cine_registred_dataset_path = constants.CINE_ADAPTED_DATA_PATH_ROI_DTC
            self.de_registred_dataset_path = constants.DE_ADAPTED_DATA_PATH_ROI_DCT
        elif self.reg_var.get() == constants.ROI_ENHANCEMENT_OBJ_DETECTION:
            self.cine_registred_dataset_path = constants.CINE_ADAPTED_DATA_PATH_ROI_EH_DCT
            self.de_registred_dataset_path = constants.DE_ADAPTED_DATA_PATH_ROI_EH_DCT
        elif self.reg_var.get() == constants.NOT_REGISTRED:
            self.cine_registred_dataset_path = constants.CINE_ADAPTED_DATA_PATH_GENERAL
            self.de_registred_dataset_path = constants.DE_ADAPTED_DATA_PATH_GENERAL

        # is generated the adapted nifti?
        is_generated_real_adapted_nifti = os.path.exists(
            os.path.join(self.cine_registred_dataset_path, "patient" + str(self.cine_combo.get()),
                         "patient" + str(self.cine_combo.get()) + "_frame01_adapted_gt.nii.gz")) and os.path.exists(
            os.path.join(self.de_registred_dataset_path, "Case_" + str(self.de_combo.get()), "Images",
                         "Case_" + str(self.de_combo.get()) + "_adapted.nii.gz"))

        is_generated_mask_adapted_nifti = os.path.exists(
            os.path.join(self.cine_registred_dataset_path, "patient" + str(self.cine_combo.get()),
                         "patient" + str(self.cine_combo.get()) + "_frame01_adapted_gt.nii.gz")) and os.path.exists(
            os.path.join(self.de_registred_dataset_path, "Case_" + str(self.de_combo.get()), "Contours",
                         "Case_P" + str(self.de_combo.get()) + "_adapted_gt.nii.gz"))

        # if radio buttons are active, load the data
        if self.var.get() == constants.REAL_IMAGE:
            if is_generated_real_adapted_nifti:
                self.load_real_image_callBack()
            else:
                self.message_label['text'] = "Error: The registred nifti file does not exist!"
                self.message_label['fg'] = constants.colorExitButton
                self.message_label.place(x=10, y=10)
        elif self.var.get() == constants.MASK:
            if is_generated_mask_adapted_nifti:
                self.load_mask_callBack()
            else:
                self.message_label['text'] = "Error: The registred nifti file does not exist!"
                self.message_label['fg'] = constants.colorExitButton
                self.message_label.place(x=10, y=10)
        elif self.var.get() == constants.OVERLAY:
            if is_generated_real_adapted_nifti and is_generated_mask_adapted_nifti:
                self.load_overlay_callBack()
            else:
                self.message_label['text'] = "Error: The registred nifti file does not exist!"
                self.message_label['fg'] = constants.colorExitButton
                self.message_label.place(x=10, y=10)
        elif self.var.get() == constants.OPPOSITE_OVERLAY:
            if is_generated_real_adapted_nifti and is_generated_mask_adapted_nifti:
                self.load_ops_overlay_callBack()
            else:
                self.message_label['text'] = "Error: The registred nifti file does not exist!"
                self.message_label['fg'] = constants.colorExitButton
                self.message_label.place(x=10, y=10)




    # browse background image
    def background_browse_callBack(self):
        # reset the images frame and its content + corresponding frame
        self.reset_corresponding_images()
        # open dialog to select a nifti file
        self.fileName = filedialog.askopenfilename(initialdir=constants.TEST, title="Select a nifti File",
                                                   filetype=(("nii.gz", "*.nii.gz"), ("nii", "*.nii")))
        if not utils.isEmpty(self.fileName):
            # load background MRI
            bg_img = nib.load(self.fileName)
            bg_data = bg_img.get_fdata()
            self.total_slices_number = bg_data.shape[2]
            # extract the slices from nifti file into the temp directory
            self.dataManager.load_slices(bg_data, constants.CINE_SAVE_PATH,
                                         constants.DO_SAVE)

            img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            # show the image in the images label frame
            image_temp = Image.open(img_fullname)
            (self.cine_width, self.cine_height, self.first_image_label) = self.show_image(self.first_image_label,
                                                                                          image_temp, 50)
            # read and show the dimensions and slice number
            slash_idx = self.fileName.rfind('/')
            self.read_dimensions(self.fileName[slash_idx + 1:], constants.CINE_TYPE)
            # enable foreground button
            self.foreground_browse_button['state'] = NORMAL

    # browse foreground image
    def foreground_browse_callBack(self):
        # remove the contents of DE MRI slices from temp directory
        utils.remove_contents(constants.DE_SAVE_PATH)
        # open dialog to select a nifti file
        self.fileName = filedialog.askopenfilename(initialdir=constants.TEST, title="Select a nifti File",
                                                   filetype=(("nii.gz", "*.nii.gz"), ("nii", "*.nii")))

        # load foreground MRI
        fg_img = nib.load(self.fileName)
        fg_data = fg_img.get_fdata()
        self.total_slices_number = fg_data.shape[2]
        # extract the slices from nifti file into the temp directory
        self.dataManager.load_slices(fg_data, constants.DE_SAVE_PATH, constants.DO_SAVE)

        # get the fullname of background and foreground slices which were extracted into temp directories
        bg_img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
        fg_img_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"

        # open both of slices
        bg_image_temp = Image.open(bg_img_fullname, 'r')
        fg_image_temp = Image.open(fg_img_fullname, 'r')

        # convert PIL.Image to cv2 and then from grayscale to RGB
        fg_img_cv2_format = cv2.cvtColor(np.asarray(Image.open(fg_img_fullname)), cv2.COLOR_RGB2BGR)
        # convert grayscale image to colored
        colored_fg_image_temp = Image.fromarray(cv2.cvtColor(utils.add_color(fg_img_cv2_format), cv2.COLOR_BGR2RGB))

        # paste the foreground on the background image
        image_temp = Image.new('RGBA', bg_image_temp.size, (0, 0, 0, 0))
        image_temp.paste(bg_image_temp, (0, 0))
        image_temp.paste(colored_fg_image_temp, (0, 0), mask=fg_image_temp)

        # show overlaid image (slice 0)
        (self.cine_width, self.cine_height, self.first_image_label) = self.show_image(self.first_image_label,
                                                                                      image_temp, 50)
        # set the mode to overlaid to check in next and previous methods
        self.overlaid_two_images = True

        slash_idx = self.fileName.rfind('/')
        self.cine_name_label['text'] += ",\n" + self.fileName[slash_idx + 1:]

    # pre-processing all of the correspondence MRIs
    def process_callBack(self):
        self.reset_message()
        # preprocess all of the cases which are defined in association.txt file
        preprocessed_total_number = self.dataManager.preprocess(self.reg_var)
        # show the successful message
        self.message_label['text'] = "Success: " + str(preprocessed_total_number) + " cases preprocessed successfully."
        self.message_label['fg'] = constants.colorSuccessFG
        self.message_label.place(x=10, y=10)

    # load the images(CINE image, DE image, overlay of both images)
    def load_real_image_callBack(self):
        # reset the general frame by selecting the radio buttons
        self.reset_general_images_by_by_radio()
        if self.cine_combo.current() > 0:
            # make the name of the CINE to load
            cine_case = os.path.join("patient" + str(self.cine_combo.get()),
                                     "patient" + str(self.cine_combo.get()) + "_frame01_adapted.nii.gz")
            self.cine_file_name = os.path.join(self.cine_registred_dataset_path, cine_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.cine_file_name, constants.CINE_SAVE_PATH)

            # make the name of slice to show
            img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            cine_image_temp = cv2.imread(img_fullname)

            # show the slice
            (self.cine_width, self.cine_height, self.first_image_label) = self.show_image(self.first_image_label,
                                                                                          cine_image_temp, 50)
            # read the dimensions value of first image
            self.read_dimensions("patient" + str(self.cine_combo.get()) + "_frame01_adapted.nii.gz",
                                 constants.CINE_TYPE)
            # ______________DE_MRI____________
            # make the name of the DE to load
            de_case = os.path.join("Case_" + str(self.de_combo.get()), "Images",
                                   "Case_" + str(self.de_combo.get()) + "_adapted.nii.gz")

            self.de_file_name = os.path.join(self.de_registred_dataset_path, de_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.de_file_name, constants.DE_SAVE_PATH)

            # make the name of slice to show
            img_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            de_image_temp = cv2.imread(img_fullname)
            # utils.get_statstics(de_image_temp, str(self.de_combo.get()) + "-" + str(self.de_slice_number))
            # show the slice
            (self.de_width, self.de_height, self.second_image_label) = self.show_image(self.second_image_label,
                                                                                       de_image_temp,
                                                                                       self.cine_width + 150)
            # read the dimensions value of second image
            self.read_dimensions("Case_" + str(self.de_combo.get()) + "_adapted.nii.gz", constants.DE_TYPE)

            # registration results
            first_img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            second_img_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"

            first_img = sitk.ReadImage(first_img_fullname, sitk.sitkFloat32)
            second_img = sitk.ReadImage(second_img_fullname, sitk.sitkFloat32)

            simg_1 = sitk.Cast(sitk.RescaleIntensity(first_img), sitk.sitkUInt8)
            simg_2 = sitk.Cast(sitk.RescaleIntensity(second_img), sitk.sitkUInt8)
            registration_temp = sitk.Compose(simg_1, simg_2, simg_1 // 2. + simg_2 // 2.)
            self.show_registration_image(registration_temp)

    # load the masks(CINE mask, DE mask, overlay of both masks)
    def load_mask_callBack(self):
        # reset the general frame by selecting the radio buttons
        self.reset_general_images_by_by_radio()
        if self.cine_combo.current() > 0:
            # make the name of the CINE mask to load
            cine_case = os.path.join("patient" + str(self.cine_combo.get()),
                                     "patient" + str(self.cine_combo.get()) + "_frame01_adapted_gt.nii.gz")
            self.cine_file_name = os.path.join(self.cine_registred_dataset_path, cine_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.cine_file_name, constants.CINE_MASK_SAVE_PATH)

            # make the name of slice to show
            img_fullname = constants.CINE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            # convert grayscale to color image
            cine_image_temp = utils.add_color(cv2.imread(img_fullname))
            # show the slice
            (self.cine_width, self.cine_height, self.first_image_label) = self.show_image(self.first_image_label,
                                                                                          cine_image_temp, 50)
            # read the dimensions value of first image
            self.read_dimensions("patient" + str(self.cine_combo.get()) + "_frame01_adapted_gt.nii.gz",
                                 constants.CINE_TYPE)
            # ______________DE_MASK_MRI____________
            # make the name of the DE mask to load
            de_case = os.path.join("Case_" + str(self.de_combo.get()), "Contours",
                                   "Case_P" + str(self.de_combo.get()) + "_adapted_gt.nii.gz")
            self.de_file_name = os.path.join(self.de_registred_dataset_path, de_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.de_file_name, constants.DE_MASK_SAVE_PATH)

            # make the name of slice to show
            img_fullname = constants.DE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            # convert grayscale to color image
            de_image_temp = utils.add_color(cv2.imread(img_fullname))
            # show the slice
            (self.de_width, self.de_height, self.second_image_label) = self.show_image(self.second_image_label,
                                                                                       de_image_temp,
                                                                                       self.cine_width + 150)
            # read the dimensions value of second image
            self.read_dimensions("Case_P" + str(self.de_combo.get()) + "_adapted_gt.nii.gz", constants.DE_TYPE)

            # registration results
            first_img_fullname = constants.CINE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            second_img_fullname = constants.DE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"

            # compute the dice coefficient between first and last slices in CINE and DE
            dice_score = utils.compute_avg_dice_score(cv2.imread(first_img_fullname), cv2.imread(second_img_fullname))
            hausdorff = utils.compute_avg_hausdorff(cv2.imread(first_img_fullname), cv2.imread(second_img_fullname))
            # utils.get_dice_hasudorff_distances(dice_score,hausdorff,str(self.cine_combo.get()) + "-" + str(self.cine_slice_number))

            self.average_dice_label['text'] = "Average Dice Score: " + str(round(dice_score[2], 3))
            self.epi_dice_label['text'] = "Left Ventricle Dice Score: " + str(round(dice_score[0], 3))
            self.myo_dice_label['text'] = "Myocardium Dice Score: " + str(round(dice_score[1], 3))
            self.average_hausdorff_label['text'] = "Average Hausdorff Distance: " + str(round(hausdorff[2], 3))
            self.epi_hausdorff_label['text'] = "Left Ventricle Hausdorff Distance: " + str(round(hausdorff[0], 3))
            self.myo_hausdorff_label['text'] = "Myocardium Hausdorff Distance: " + str(round(hausdorff[1], 3))

            first_temp = cv2.imread(first_img_fullname)
            second_temp = utils.add_color(cv2.imread(second_img_fullname))
            first_image_temp = cv2.addWeighted(first_temp, 0.5, second_temp, 0.5, 0)
            # show second image label frame
            self.third_images_label_frame.place(x=1215, y=60)
            self.show_image(self.third_image_label, first_image_temp, 50)

    # load overlay (CINE image + CINE mask, DE image + DE mask)
    def load_overlay_callBack(self):
        # reset the general frame by selecting the radio buttons
        self.reset_general_images_by_by_radio()
        if self.cine_combo.current() > 0:
            # make the name of the CINE to load
            cine_case = os.path.join("patient" + str(self.cine_combo.get()),
                                     "patient" + str(self.cine_combo.get()) + "_frame01_adapted.nii.gz")
            self.cine_file_name = os.path.join(self.cine_registred_dataset_path, cine_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.cine_file_name, constants.CINE_SAVE_PATH)

            # make the name of the CINE mask to load
            cine_case = os.path.join("patient" + str(self.cine_combo.get()),
                                     "patient" + str(self.cine_combo.get()) + "_frame01_adapted_gt.nii.gz")
            self.cine_file_name = os.path.join(self.cine_registred_dataset_path, cine_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.cine_file_name, constants.CINE_MASK_SAVE_PATH)

            # make the name of real and mask slices to show
            img_real_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            img_mask_fullname = constants.CINE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"

            cine_real_image_temp = cv2.imread(img_real_fullname)
            # convert grayscale to color image
            cine_mask_image_temp = utils.add_color(cv2.imread(img_mask_fullname))
            cine_image_temp = cv2.addWeighted(cine_real_image_temp, 0.4, cine_mask_image_temp, 0.6, 0)
            # show the slice
            (self.cine_width, self.cine_height, self.first_image_label) = self.show_image(self.first_image_label,
                                                                                          cine_image_temp, 50)
            # read the dimensions value of first image
            self.read_dimensions("patient" + str(self.cine_combo.get()) + "_frame01_adapted.nii.gz,\npatient" + str(
                self.cine_combo.get()) + "_frame01_adapted_gt.nii.gz", constants.CINE_TYPE)

            # ______________DE_OVERLAID_MRI____________

            # make the name of the DE to load
            de_case = os.path.join("Case_" + str(self.de_combo.get()), "Images",
                                   "Case_" + str(self.de_combo.get()) + "_adapted.nii.gz")

            self.de_file_name = os.path.join(self.de_registred_dataset_path, de_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.de_file_name, constants.DE_SAVE_PATH)

            # make the name of the DE mask to load
            de_case = os.path.join("Case_" + str(self.de_combo.get()), "Contours",
                                   "Case_P" + str(self.de_combo.get()) + "_adapted_gt.nii.gz")
            self.de_file_name = os.path.join(self.de_registred_dataset_path, de_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.de_file_name, constants.DE_MASK_SAVE_PATH)

            # make the name of real and mask slices to show
            img_real_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
            img_mask_fullname = constants.DE_MASK_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"

            de_real_image_temp = cv2.imread(img_real_fullname)
            # convert grayscale to color image
            de_mask_image_temp = utils.add_color(cv2.imread(img_mask_fullname))
            de_image_temp = cv2.addWeighted(de_real_image_temp, 0.5, de_mask_image_temp, 0.5, 0)
            # show the slice
            (self.de_width, self.de_height, self.second_image_label) = self.show_image(self.second_image_label,
                                                                                       de_image_temp,
                                                                                       self.cine_width + 150)
            # read the dimensions value of second image
            self.read_dimensions(
                "Case_" + str(self.de_combo.get()) + "_adapted.nii.gz,\nCase_P" + str(
                    self.de_combo.get()) + "_adapted_gt.nii.gz", constants.DE_TYPE)

    # load opposite overlay (CINE image + DE mask, DE image + CINE mask)
    def load_ops_overlay_callBack(self):
        # reset the general frame by selecting the radio buttons
        self.reset_general_images_by_by_radio()
        if self.cine_combo.current() > 0:
            # make the name of the CINE to load
            cine_case = os.path.join("patient" + str(self.cine_combo.get()),
                                     "patient" + str(self.cine_combo.get()) + "_frame01_adapted.nii.gz")
            self.cine_file_name = os.path.join(self.cine_registred_dataset_path, cine_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.cine_file_name, constants.CINE_SAVE_PATH)

            # make the name of the DE mask to load
            de_case = os.path.join("Case_" + str(self.de_combo.get()), "Contours",
                                   "Case_P" + str(self.de_combo.get()) + "_adapted_gt.nii.gz")
            self.de_file_name = os.path.join(self.de_registred_dataset_path, de_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.de_file_name, constants.DE_MASK_SAVE_PATH)

            # make the name of real and mask slices to show
            img_real_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
            img_mask_fullname = constants.DE_MASK_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"

            cine_real_image_temp = cv2.imread(img_real_fullname)
            # convert grayscale to color image
            cine_mask_image_temp = utils.add_color(cv2.imread(img_mask_fullname))
            cine_image_temp = cv2.addWeighted(cine_real_image_temp, 0.4, cine_mask_image_temp, 0.6, 0)
            # show the slice
            (self.cine_width, self.cine_height, self.first_image_label) = self.show_image(self.first_image_label,
                                                                                          cine_image_temp, 50)
            # read the dimensions value of first image
            self.read_dimensions("patient" + str(self.cine_combo.get()) + "_frame01_adapted.nii.gz,\nCase_P" + str(
                self.de_combo.get()) + "_adapted_gt.nii.gz", constants.CINE_TYPE)

            # ______________DE_OVERLAID_MRI____________

            # make the name of the DE to load
            de_case = os.path.join("Case_" + str(self.de_combo.get()), "Images",
                                   "Case_" + str(self.de_combo.get()) + "_adapted.nii.gz")

            self.de_file_name = os.path.join(self.de_registred_dataset_path, de_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.de_file_name, constants.DE_SAVE_PATH)

            # make the name of the DE mask to load
            cine_case = os.path.join("patient" + str(self.cine_combo.get()),
                                     "patient" + str(self.cine_combo.get()) + "_frame01_adapted_gt.nii.gz")
            self.cine_file_name = os.path.join(self.cine_registred_dataset_path, cine_case)

            # extract nifti file to its slices in the temp path
            self.extract_data(self.cine_file_name, constants.CINE_MASK_SAVE_PATH)

            # make the name of real and mask slices to show
            img_real_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
            img_mask_fullname = constants.CINE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"

            de_real_image_temp = cv2.imread(img_real_fullname)
            # convert grayscale to color image
            de_mask_image_temp = utils.add_color(cv2.imread(img_mask_fullname))
            de_image_temp = cv2.addWeighted(de_real_image_temp, 0.5, de_mask_image_temp, 0.5, 0)
            # show the slice
            (self.de_width, self.de_height, self.second_image_label) = self.show_image(self.second_image_label,
                                                                                       de_image_temp,
                                                                                       self.cine_width + 150)
            # read the dimensions value of second image
            self.read_dimensions("Case_" + str(self.de_combo.get()) + "_adapted.nii.gz,\npatient" + str(
                self.cine_combo.get()) + "_frame01_adapted_gt.nii.gz", constants.DE_TYPE)

    # close the session
    def exit_callBack(self):
        self.destroy()

    # update de combobox based on selected option of cine combobox and correspondence MRI defined in association.txt
    def update_de_combobox(self, event):
        # reset the images frame and its content + general frame
        self.reset_general_images_by_combo()

        # set the option of de combo based on selected value in cine combo
        self.de_combo.current(self.cine_cases.index(self.cine_combo.get()))

        # if radio buttons are active, load the data
        if self.var.get() == constants.REAL_IMAGE:
            self.load_real_image_callBack()
        elif self.var.get() == constants.MASK:
            self.load_mask_callBack()
        elif self.var.get() == constants.OVERLAY:
            self.load_overlay_callBack()
        elif self.var.get() == constants.OPPOSITE_OVERLAY:
            self.load_ops_overlay_callBack()

    # update cine combobox based on selected option of de combobox and correspondence MRI defined in association.txt
    def update_cine_combobox(self, event):
        # reset the images frame and its content + general frame
        self.reset_general_images_by_combo()

        # set the option of cine combo based on selected value in de combo
        self.cine_combo.current(self.de_cases.index(self.de_combo.get()))

        # if radio buttons are active, load the data
        if self.var.get() == constants.REAL_IMAGE:
            self.load_real_image_callBack()
        elif self.var.get() == constants.MASK:
            self.load_mask_callBack()
        elif self.var.get() == constants.OVERLAY:
            self.load_overlay_callBack()
        elif self.var.get() == constants.OPPOSITE_OVERLAY:
            self.load_ops_overlay_callBack()

    # extract file into the temporary directory
    def extract_data(self, file_name, temp_path):
        # load the nifti file
        img = nib.load(file_name)
        data = img.get_fdata()
        self.total_slices_number = data.shape[2]
        # load nifti and extract it
        self.dataManager.load_slices(data, temp_path, constants.DO_SAVE)

    # reset message label
    def reset_message (self):
        self.message_label['text'] = ""
        self.message_label.place(x=10, y=10)

    # reset the images frame and its content + corresponding frame
    def reset_corresponding_images(self):
        self.cine_slice_number = 0
        self.de_slice_number = 0
        self.is_csp_mode = False

        self.var.set(0)
        self.cine_combo.current(0)
        self.de_combo.current(0)

        self.real_image_radio['state'] = DISABLED
        self.mask_radio['state'] = DISABLED
        self.overlay_radio['state'] = DISABLED
        self.opposite_overlay_radio['state'] = DISABLED

        self.first_image_label.config(image='', bg=constants.colorActionBg)
        self.second_image_label.config(image='', bg=constants.colorActionBg)
        self.third_image_label.config(image='', bg=constants.colorActionBg)
        self.second_images_label_frame.place_forget()
        self.third_images_label_frame.place_forget()

        self.remove_contents(constants.CINE_SAVE_PATH)
        self.remove_contents(constants.CINE_MASK_SAVE_PATH)
        self.remove_contents(constants.DE_SAVE_PATH)
        self.remove_contents(constants.DE_MASK_SAVE_PATH)

        self.cine_name_label['text'] = ''
        self.cine_dim_label['text'] = ''
        self.cine_slice_number_label['text'] = ''

        self.de_name_label['text'] = ''
        self.de_dim_label['text'] = ''
        self.de_slice_number_label['text'] = ''

        self.reg_name_label['text'] = ''
        self.reg_dim_label['text'] = ''
        self.reg_slice_number_label['text'] = ''
        # reset the slide bar from start
        self.slide_bar.set(self.cine_slice_number + 1)

        self.slide_bar_label.place_forget()
        self.slide_bar.place_forget()

    # reset the images frame and its content + general frame by selecting the comboboxes
    def reset_general_images_by_combo(self):
        self.cine_slice_number = 0
        self.de_slice_number = 0
        self.is_csp_mode = True

        self.foreground_browse_button['state'] = DISABLED

        self.first_image_label.config(image='', bg=constants.colorActionBg)

        self.cine_name_label['text'] = ''
        self.cine_dim_label['text'] = ''
        self.cine_slice_number_label['text'] = ''

        self.de_name_label['text'] = ''
        self.de_dim_label['text'] = ''
        self.de_slice_number_label['text'] = ''

        self.reg_name_label['text'] = ''
        self.reg_dim_label['text'] = ''
        self.reg_slice_number_label['text'] = ''
        self.average_dice_label['text'] = ''
        self.epi_dice_label['text'] = ''
        self.myo_dice_label['text'] = ''
        self.average_hausdorff_label['text'] = ''
        self.epi_hausdorff_label['text'] = ''
        self.myo_hausdorff_label['text'] = ''

        self.real_image_radio['state'] = NORMAL
        self.mask_radio['state'] = NORMAL
        self.overlay_radio['state'] = NORMAL
        self.opposite_overlay_radio['state'] = NORMAL
        # show second and third image label frame
        self.second_images_label_frame.place(x=880, y=60)
        self.third_images_label_frame.place(x=1215, y=60)
        # reset the slide bar from start
        self.slide_bar.set(self.cine_slice_number + 1)

    # reset the images frame and its content + general frame by selecting the radiobuttons
    def reset_general_images_by_by_radio(self):
        self.is_csp_mode = True
        self.overlaid_two_images = False

        self.foreground_browse_button['state'] = DISABLED

        self.remove_contents(constants.CINE_SAVE_PATH)
        self.remove_contents(constants.CINE_MASK_SAVE_PATH)
        self.remove_contents(constants.DE_SAVE_PATH)
        self.remove_contents(constants.DE_MASK_SAVE_PATH)

        self.second_image_label.config(image='', bg=constants.colorActionBg)
        self.third_image_label.config(image='', bg=constants.colorActionBg)

        if self.var.get() == constants.OVERLAY or self.var.get() == constants.OPPOSITE_OVERLAY:
            self.third_image_label.config(image='', bg=constants.colorActionBg)
            self.third_images_label_frame.place_forget()

    # remove the contents of directories
    def remove_contents(self, path):
        files = os.listdir(path)
        for f in files:
            os.remove(os.path.join(path, f))

    # show image in the label
    def show_image(self, image_label, image_temp, x_pos):
        ##
        (h, w) = image_temp.shape[:2]
        (cX, cY) = (w / 2, h / 2)
        # rotate our image by 45 degrees
        M = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
        image_temp = cv2.warpAffine(image_temp, M, (w, h))
        ##
        if image_temp is not None:
            if isinstance(image_temp, np.ndarray):
                image_array = Image.fromarray(image_temp)
            else:
                image_array = image_temp



            image = ImageTk.PhotoImage(image=image_array)
            width = image.width()
            height = image.height()

            image_label.config(image='', bg=constants.colorActionBg)
            image_label["image"] = image

            image_label.img = image
            image_label.place(x=15, y=85)
            return width, height, image_label

    # show composition of fix and registered image
    def show_registration_image(self, image_temp):

        if image_temp is not None:
            self.third_image_label.config(image='', bg=constants.colorActionBg)
            ##
            x = sitk.GetArrayFromImage(image_temp)
            (h, w) = x.shape[:2]
            (cX, cY) = (w / 2, h / 2)
            # rotate our image by 45 degrees
            M = cv2.getRotationMatrix2D((128, 131.5), 180, 1.0)
            x = cv2.warpAffine(x, M, (w, h))
            ##

            final_img = ImageTk.PhotoImage(image=Image.fromarray(x))
            self.third_image_label["image"] = final_img
            self.third_image_label.img = final_img
            # show second image label frame
            self.third_images_label_frame.place(x=1215, y=60)
            self.third_image_label.place(x=15, y=85)

    # read and set the filename ,dimension info and slice number of images
    def read_dimensions(self, fileName, mri_type):
        if mri_type == constants.CINE_TYPE:
            self.cine_name_label['text'] = "File:  " + fileName + "\n(Registered File)"
            self.cine_dim_label['text'] = "Dimensions:    x: " + str(self.cine_width) + "    y: " + str(
                self.cine_height) + "    z: " + str(self.total_slices_number)
            self.cine_slice_number_label['text'] = "#Slice: " + str(self.cine_slice_number + 1)

            self.cine_name_label["justify"] = LEFT

            self.cine_name_label.place(x=20, y=5)
            self.cine_dim_label.place(x=20, y=50)
            self.cine_slice_number_label.place(x=20, y=65)
        else:
            self.de_name_label['text'] = "File:  " + fileName
            self.de_dim_label['text'] = "Dimensions:    x: " + str(self.de_width) + "    y: " + str(
                self.de_height) + "    z: " + str(self.total_slices_number)
            self.de_slice_number_label['text'] = "#Slice: " + str(self.de_slice_number + 1)

            self.de_name_label["justify"] = LEFT

            self.de_name_label.place(x=20, y=5)
            self.de_dim_label.place(x=20, y=50)
            self.de_slice_number_label.place(x=20, y=65)

        if self.var.get() is not None and self.var.get() == constants.REAL_IMAGE or self.var.get() == constants.MASK:
            self.reg_name_label['text'] = "File: Overlay of registered CINE-MRI and DE-MRI"
            self.reg_dim_label['text'] = "Dimensions:    x: " + str(self.cine_width) + "    y: " + str(
                self.cine_height) + "    z: " + str(self.total_slices_number)
            self.reg_slice_number_label['text'] = "#Slice: " + str(self.de_slice_number + 1)

            self.reg_name_label.place(x=20, y=5)
            self.reg_dim_label.place(x=20, y=50)
            self.reg_slice_number_label.place(x=20, y=65)

            self.average_dice_label.place(x=20, y=self.cine_height + 100)
            self.epi_dice_label.place(x=20, y=self.cine_height + 120)
            self.myo_dice_label.place(x=20, y=self.cine_height + 140)
            self.average_hausdorff_label.place(x=20, y=self.cine_height + 160)
            self.epi_hausdorff_label.place(x=20, y=self.cine_height + 180)
            self.myo_hausdorff_label.place(x=20, y=self.cine_height + 200)

        self.slide_bar["to"] = self.total_slices_number
        self.slide_bar_label.place(x=550, y=23)
        self.slide_bar.place(x=600, y=5)

    # sliding between slices
    def slide(self, var):
        self.cine_slice_number = self.slide_bar.get() - 1
        self.de_slice_number = self.slide_bar.get() - 1
        if self.is_csp_mode:
            if self.var.get() == constants.REAL_IMAGE:
                img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                bg_image_temp = cv2.imread(img_fullname)

                # registration results
                first_img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                second_img_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                # utils.get_statstics(cv2.imread(second_img_fullname), str(self.de_combo.get()) + "-" + str(self.de_slice_number))


                first_img = sitk.ReadImage(first_img_fullname, sitk.sitkFloat32)
                second_img = sitk.ReadImage(second_img_fullname, sitk.sitkFloat32)

                first_simg = sitk.Cast(sitk.RescaleIntensity(first_img), sitk.sitkUInt8)
                second_simg = sitk.Cast(sitk.RescaleIntensity(second_img), sitk.sitkUInt8)
                registration_temp = sitk.Compose(first_simg, second_simg, first_simg // 2. + second_simg // 2.)
                self.show_registration_image(registration_temp)

            if self.is_csp_mode and self.var.get() == constants.MASK:
                img_fullname = constants.CINE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                bg_image_temp = utils.add_color(cv2.imread(img_fullname))
                #
                first_img_fullname = constants.CINE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                second_img_fullname = constants.DE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"

                # compute the dice coefficient between slices in CINE and DE
                dice_score = utils.compute_avg_dice_score(cv2.imread(first_img_fullname),
                                                          cv2.imread(second_img_fullname))
                hausdorff = utils.compute_avg_hausdorff(cv2.imread(first_img_fullname),
                                                             cv2.imread(second_img_fullname))
                # utils.get_dice_hasudorff_distances(dice_score, hausdorff,
                #                                    str(self.cine_combo.get()) + "-" + str(self.cine_slice_number))

                self.average_dice_label['text'] = "Average Dice Score: " + str(round(dice_score[2], 3))
                self.epi_dice_label['text'] = "Left Ventricle Dice Score: " + str(round(dice_score[0], 3))
                self.myo_dice_label['text'] = "Myocardium Dice Score: " + str(round(dice_score[1], 3))
                self.average_hausdorff_label['text'] = "Hausdorff Distance: " + str(round(hausdorff[2], 3))
                self.epi_hausdorff_label['text'] = "Left Ventricle Hausdorff Distance: " + str(round(hausdorff[0], 3))
                self.myo_hausdorff_label['text'] = "Myocardium Hausdorff Distance: " + str(round(hausdorff[1], 3))

                first_temp = cv2.imread(first_img_fullname)
                second_temp = utils.add_color(cv2.imread(second_img_fullname))
                first_image_temp = cv2.addWeighted(first_temp, 0.5, second_temp, 0.5, 0)
                self.show_image(self.third_image_label, first_image_temp, 50)

            if self.is_csp_mode and self.var.get() == constants.OVERLAY:
                bg_img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                fg_img_fullname = constants.CINE_MASK_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                bg_temp = cv2.imread(bg_img_fullname)
                fg_temp = utils.add_color(cv2.imread(fg_img_fullname))
                bg_image_temp = cv2.addWeighted(bg_temp, 0.5, fg_temp, 0.5, 0)

            if self.is_csp_mode and self.var.get() == constants.OPPOSITE_OVERLAY:
                bg_img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                fg_img_fullname = constants.DE_MASK_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
                bg_temp = cv2.imread(bg_img_fullname)
                fg_temp = utils.add_color(cv2.imread(fg_img_fullname))
                bg_image_temp = cv2.addWeighted(bg_temp, 0.5, fg_temp, 0.5, 0)
        else:
            if not self.overlaid_two_images:
                img_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                bg_image_temp = cv2.imread(img_fullname)

            if self.overlaid_two_images:
                bg_fullname = constants.CINE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                fg_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.cine_slice_number) + ".png"
                bg_temp = Image.open(bg_fullname, 'r')
                fg_temp = Image.open(fg_fullname, 'r')

                # convert PIL.Image to cv2 and then from grayscale to RGB
                fg_img_cv2 = cv2.cvtColor(np.asarray(Image.open(fg_fullname)), cv2.COLOR_RGB2BGR)
                colored_second_image_temp = Image.fromarray(
                    cv2.cvtColor(utils.add_color(fg_img_cv2), cv2.COLOR_BGR2RGB))

                bg_image_temp = Image.new('RGBA', bg_temp.size, (0, 0, 0, 0))
                bg_image_temp.paste(bg_temp, (0, 0))
                bg_image_temp.paste(colored_second_image_temp, (0, 0), mask=fg_temp)
        (self.cine_width, self.cine_height, self.first_image_label) = self.show_image(self.first_image_label,
                                                                                      bg_image_temp, 50)
        # if self.cine_image_temp is not None:
        self.cine_slice_number_label['text'] = "#Slice: " + str(self.cine_slice_number + 1)

        # ______________DE_MRI____________

        if self.is_csp_mode:
            if self.var.get() == constants.REAL_IMAGE:
                img_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
                de_image_temp = cv2.imread(img_fullname)

            if self.var.get() == constants.MASK:
                img_fullname = constants.DE_MASK_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
                de_image_temp = utils.add_color(cv2.imread(img_fullname))

            if self.var.get() == constants.OVERLAY:
                img_real_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
                img_mask_fullname = constants.DE_MASK_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
                de_real_image_temp = cv2.imread(img_real_fullname)
                de_mask_image_temp = utils.add_color(cv2.imread(img_mask_fullname))
                de_image_temp = cv2.addWeighted(de_real_image_temp, 0.5, de_mask_image_temp, 0.5, 0)

            if self.var.get() == constants.OPPOSITE_OVERLAY:
                img_real_fullname = constants.DE_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
                img_mask_fullname = constants.CINE_MASK_SAVE_PATH + "/slice_" + str(self.de_slice_number) + ".png"
                de_real_image_temp = cv2.imread(img_real_fullname)
                de_mask_image_temp = utils.add_color(cv2.imread(img_mask_fullname))
                de_image_temp = cv2.addWeighted(de_real_image_temp, 0.5, de_mask_image_temp, 0.5, 0)

            (self.cine_width, self.cine_height, self.second_image_label) = self.show_image(
                self.second_image_label,
                de_image_temp, self.cine_width + 150)
            # if self.cine_image_temp is not None:
            self.de_slice_number_label['text'] = "#Slice: " + str(self.de_slice_number + 1)
            self.reg_slice_number_label['text'] = "#Slice: " + str(self.de_slice_number + 1)

    # visual method (style)
    def set_stylies(self):
        # styles
        self.font_style_header_up = Font(family="ARIAL", size=18, weight=BOLD)
        self.fontStyleHeader_down = Font(family="ARIAL", size=11, weight=BOLD)
        self.fontStyleTabHeader_down = Font(family="ARIAL", size=10, weight=BOLD)
        self.fontStyleContent_labelFrame = Font(family="ARIAL", size=9)
        self.fontStyleButton_labelFrame = Font(family="ARIAL", size=9, weight=BOLD)
        self.fontStyleLabelFooter = Font(family="ARIAL", size=8)

        self.style_tabControl = Style()
        self.style_tabControl.configure('.', background=constants.colorContent)
        self.style_tabControl.configure('TNotebook', background=constants.colorActionBg, tabmargins=[5, 5, 0, 0])
        self.style_tabControl.map("TNotebook.Tab", foreground=[("selected", constants.colorSelected)])
        self.style_tabControl.configure('TNotebook.Tab', padding=[10, 4], font=('ARIAL', '13', 'bold'),
                                        foreground=constants.colorUnSelected)

        self.style_radioButton = Style()
        self.style_radioButton.configure('Wild.TRadiobutton', background="SystemWindow",
                                         foreground=constants.colorSelected,
                                         font=self.fontStyleContent_labelFrame)

    # visual method (component position)
    def set_positions(self):
        # header
        self.header_up.place(x=50, y=50)
        self.header_down.place(x=50, y=50)
        self.header_up.pack(fill="x")
        self.header_down.pack(fill="x")
        # icon
        self.imageIcon_label.place(x=0, y=3)
        # tab
        self.tabControl.pack(expand=1, fill="both")

        # loading frames (corresponding and general)
        self.loading_csp_label_frame.place(x=10, y=110)
        self.loading_gnl_label_frame.place(x=10, y=310)
        # loading any data components
        self.background_data_label.place(x=0, y=10)
        self.background_browse_button.place(x=120, y=10)

        self.foreground_data_label.place(x=320, y=10)
        self.foreground_browse_button.place(x=440, y=10)

        # images frame
        self.first_images_label_frame.place(x=545, y=60)
        self.second_images_label_frame.place(x=880, y=60)
        self.third_images_label_frame.place(x=1215, y=60)

        self.preprocess_info_label.place(x=10, y=40)
        # registration options
        self.registration_label.place(x=0, y=5)
        # radio button
        self.whole_image_radio.place(x=100, y=5)
        self.ROI_radio.place(x=300, y=5)
        self.ROI_obj_detection_radio.place(x=100, y=25)
        # self.ROI_enhancement_obj_detection_radio.place(x=300, y=25)
        # self.not_registred_radio.place(x=100, y=45)
        self.not_registred_radio.place(x=300, y=25)

        self.cine_label.place(x=0, y=85)
        self.de_label.place(x=320, y=85)
        self.cine_combo.place(x=60, y=85)
        self.de_combo.place(x=370, y=85)

        self.loading_options_label.place(x=0, y=125)
        # radio button
        self.real_image_radio.place(x=100, y=125)
        self.mask_radio.place(x=300, y=125)
        self.overlay_radio.place(x=100, y=145)
        self.opposite_overlay_radio.place(x=300, y=145)
        # process_button
        self.process_button.place(x=10, y=70)

        self.footPage_label.pack(side=BOTTOM, anchor="sw")
        self.exit_button.pack(side=BOTTOM, anchor="ne")


root = Root()
root.title("Final Thesis")
root.state('zoomed')
root.mainloop()
