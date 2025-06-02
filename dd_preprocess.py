#!/usr/bin/python3
"""
Preprocesses images in 2 ways:

1) Processes images to meet Transkribus upload requirements
2) Processes images which meet Transkribus upload requirements to improve readability for
OCR/HTR/field models.

Usage instructions:

Run the script from the command line once you have navigated to the location of this python file as follows...

python dd_preprocess.py path/to/source/directory path/to/destination/directory

* Source directory path: The path to the folder which stores the images you want to preprocess
* Target directory path: The path to the folder which will store the preprocessed images - this doesn't have to exist
yet, it just needs to include the desired path to/name for the folder

After these, you can optionally include the following flags:

* --k_val [float]/ -k : Modify the K-value used during Sauvola binarisation (default: 0.14)
* --window_size [int]/ -w : Modify the window size used during Sauvola binarisation.
                            Should not be an even value (default: 21)
* --contrast_enhance / -ce : Use flag if you want to contrast stretch and enhance contrast of images within
                             pipeline. Default is not to use this as it tends to introduce speckling.
* --basic_only / -b : Use flag if you only want to meet basic Transkribus upload requirements
                      and do not want to do further preprocessing like binarisation (e.g. for evaluation
                      purposes). If you use this flag, the other optional flags are irrelevant as they apply
                      to the further preprocessing pipeline.

For example:

You could use the following command to adjust the k-value and window size used during binarisation to
alter the quality of images outputted -

python dd_preprocess.py path/to/source/directory path/to/destination/directory --k_val 0.22 --window_size 301


You could use the following command to only meet the basic Transkribus image upload requirements
(e.g. file size, image format) and not perform further preprocessing like binarisation -

python dd_preprocess.py path/to/source/directory path/to/destination/directory --basic_only

"""

import os  # Deals with path names
import argparse  # Takes arguments from command line
from PIL import Image  # For image preprocessing
from tqdm import tqdm  # For progress loading bar
import cv2  # For image preprocessing
from skimage import filters, util, color  # For image preprocessing
import numpy as np  # For working with image data
from scipy import ndimage  # For working with image data


bytes_in_mb = 1000000  # the number of bytes in a megabyte
max_image_mbs = 10

# Determine maximum allowed image size (according to Transkribus upload requirements)
max_image_bytes = max_image_mbs * bytes_in_mb


def meet_upload_reqs(src_image_path, dest_image_path, basic_only):
    """
    Performs the first preprocessing step, ensuring images meet basic upload requirements and are reasonable
    dimensions for OCR/HTR.
    :param src_image_path: (str) Filepath to an individual image to be preprocessed.
    :param dest_image_path: (str) Filepath at which to save the preprocessed source image.
    :param basic_only: (bool) If true, user only wishes for this first (basic) preprocessing step to be
    performed, and no further steps like binarisation in function preprocess_image. In this case,
    image is compressed at function end. Otherwise, image is compressed at end of preprocess_image.
    """
    try:
        # Read the image
        img = Image.open(src_image_path)

        # Convert image to JPG
        img = img.convert("RGB")

        # Resize image (dimensions) - check image width < 2500 then resize by a determined factor
        # 2500 is max desired dimension for resize image
        # resizing code inspired by Medium user DLMade:
        # https://dlmade.medium.com/image-preprocessing-before-ocr-76f7047534a5
        max_dimension = 2500

        # calculate resizing factor and use to resize (increase)
        # if image is smaller than desired max dim
        factor = max(1, int(max_dimension / max(img.size)))
        if factor > 1:
            # image_size[0] = width, [1] = height
            size = int(factor * img.size[0]), int(factor * img.size[1])
            img = img.resize(size, resample=Image.LANCZOS)

        # Write the image and set desired DPI (300)
        img.save(dest_image_path, dpi=(300, 300))

        if basic_only:
            # If user only requires preparation for upload to Transkribus but no further preprocessing, perform
            # any necessary compression within this function. Otherwise, perform at end of preprocess_image function.

            img_size_bytes = os.path.getsize(dest_image_path)

            # If image is already smaller than target size, return the image
            if img_size_bytes <= max_image_bytes:
                pass
            else:
                # If image is larger than max allowed size, compress until allowable size
                compress_under_size(max_image_bytes, dest_image_path)

    except Exception as image_prep_error:
        print(f"Error preprocessing image {file} when running meet_upload_reqs function: {image_prep_error}")


def preprocess_image(src_image_path, dest_image_path, contrast_enhance, k_val, window_size):
    """
    Performs the second preprocessing step to prepare images for more accurate OCR/HTR. Includes: Greyscaling,
    denoising, (optional) constrast stretching and contrast enhancement, Sauvola binarisation, and deskewing.
    :param src_image_path: (str) Filepath to an individual image to be preprocessed.
    :param dest_image_path: (str) Filepath at which to save the preprocessed source image.
    :param contrast_enhance: (bool) If true, performs contrast stretching and contrast enhancement on given image
     Default is not to use this as it tends to introduce speckling.
    :param k_val: (float) The K-value used during Sauvola binarisation (default: 0.14)
    :param window_size: (int) The window size used during Sauvola binarisation. Should not be an even value
    (default: 21)
    """
    transk_image_extensions = ['.pdf', '.jpg', '.png']  # file types allowed by Transkribus

    try:
        # Check if the file is an image (any file with an image extension)
        if any(src_image_path.lower().endswith(image_ext) for image_ext in transk_image_extensions):
            # Read the image
            image = cv2.imread(src_image_path)
            if image is None:
                print(f"Error: Image not found or cannot be read: {src_image_path}")
                pass

            # Greyscale the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Denoise image: Fast non-local means denoising (method for greyscale images):
            image = cv2.fastNlMeansDenoising(image, None, h=10,
                                             templateWindowSize=7,
                                             searchWindowSize=21)

            # Enhance contrast (optional): Normalisation (contrast stretching) +
            # adaptive histogram equalization (CLAHE)
            if contrast_enhance:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                image = clahe.apply(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))

            # Convert between opencv's BGR format and skimage's RGB format for numpy arrays representing images
            # Avoids overhead of writing to file with cv2 then reopening with skimage
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = color.rgb2gray(image)  # Convert to greyscale with skimage for quicker processing

            # Binarise: Sauvola (local) thresholding
            sauvola_threshold = filters.threshold_sauvola(image, window_size=window_size, k=k_val)
            image = image > sauvola_threshold

            # Write image to path
            image = util.img_as_ubyte(image)  # Convert boolean image to uint8

            # Skew correction: Projection Profiling method from Susmith Reddy
            # https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7
            rotate_image(image)

            # If image is already smaller than target size, return the image
            img_size_bytes = os.path.getsize(dest_image_path)
            if img_size_bytes <= max_image_bytes:
                pass
            else:
                # If image is larger than max allowed size, compress until allowable size
                compress_under_size(max_image_bytes, dest_image_path)

    except Exception as image_preprocessing_error:
        print(f"Error preprocessing image {file} when running preprocess_image function: {image_preprocessing_error}")


def compress_under_size(desired_max_bytes, src_img_path):
    """
    Searches until function achieves an approximate compression quality value according to desired max bytes
    :param desired_max_bytes: (int) maximum desired size in bytes of image
    :param src_img_path: (str) path to the image file to be custom compressed
    """
    quality = 85  # value of 90 usually increases size

    # current image size in bytes (not megabytes)
    img_size_bytes = os.path.getsize(src_img_path)

    while img_size_bytes > desired_max_bytes or quality == 0:
        print(f"compressing further - current size: {img_size_bytes / bytes_in_mb}mb")
        if quality == 0:
            os.remove(src_img_path)
            print("Error: File cannot be compressed below this size")
            break

        compress_pic(src_img_path, quality)
        img_size_bytes = os.path.getsize(src_img_path)
        quality -= 5

    print(f"final compressed size: {img_size_bytes / bytes_in_mb}mb")


def compress_pic(src_img_path, quality):
    """
    Compressed image located at image path to given quality % while saving newly-compressed image. Helper function to
    compress_under_size.
    :param src_img_path: (str) A string to the image file to be custom compressed
    :param quality: (int) The quality to be compressed down to.
    :return: (int) size of resulting image in bytes
    """
    img = Image.open(src_img_path)

    img.save(src_img_path, "JPEG", optimize=True, quality=quality, dpi=(300,300))

    processed_size = os.path.getsize(src_img_path)

    return processed_size


def count_files_in_directory_tree(src_folder):
    """
    Counts the number of image files to be processed for use alongside tqdm progress bar.
    :param src_folder: (str) path to directory containing image files to be processed
    :return: (int) number of image files to be processed
    """
    total_file_count = 0
    for root, dirs, files in os.walk(src_folder):
        total_file_count += len(files)
    return total_file_count


def find_rotation_score(array, angle):
    """
    Returns histogram and computed rotation angle score for a given proposed angle.
    Taken from https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7.
    :return: histogram, computed rotation angle score
    """
    data = ndimage.rotate(array, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def find_rotation_angle(image):
    """
    Projection Profile method code taken from https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7.
    :param image: (np.ndarray) greyscaled and binarised/thresholded image
    :return: (float) best_rotation_angle
    """
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit + delta, delta)
    scores = []

    for angle in angles:
        hist, score = find_rotation_score(image, angle)
        scores.append(score)

    best_score = max(scores)
    best_rotation_angle = angles[scores.index(best_score)]

    return best_rotation_angle


def rotate_image(image):
    """
    Projection Profile method code taken from https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7.
    Saves image to same location it was sourced from.
    :param image: (np.ndarray) greyscaled and binarised/thresholded image
    :return: rotated image
    """
    # # Read image with Pillow
    array = np.array(image, np.uint8)

    # Find the best rotation angle
    rotation_angle = find_rotation_angle(array)

    # Rotate the image according to the best/most likely angle
    data = ndimage.rotate(array, rotation_angle, reshape=False, order=0)

    # Convert the rotated array back to PIL Image
    image = Image.fromarray(data.astype("uint8"))

    # Save rotated image
    image.save(dest_image_path)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image processing script")
    parser.add_argument("source_folder", type=str, help="Path to the source folder")
    parser.add_argument("destination_folder", type=str, help="Path to the destination folder")
    parser.add_argument("--k_val", "-k", type=float, default=0.14,
                        help="K-value in Sauvola binarisation (default: 0.14)")
    parser.add_argument("--window_size", "-w", type=int, default=21,
                        help="Window size in Sauvola binarisation. Should not be an even value (default: 21)")
    parser.add_argument("--contrast_enhance", "-ce",
                        help="Use flag if you want to contrast stretch and enhance contrast of images within "
                             "pipeline. Default is not to use this.",
                        action="store_true")
    parser.add_argument("--basic_only", "-b",
                        help="Use flag if you only want to meet basic Transkribus upload requirements "
                             "and do not want to do further preprocessing like binarisation (e.g. for evaluation "
                             "purposes).",
                        action="store_true")

    args = parser.parse_args()

    file_count = count_files_in_directory_tree(args.source_folder)

    if args.basic_only:
        if any((args.k_val != 0.14, args.window_size != 21, args.contrast_enhance)):
            print("Warning: --basic_only is specified, other arguments (k_val, window_size, contrast_enhance) "
                  "are irrelevant and will not be used.")

    # Common image file extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.ico', '.svg']

    with tqdm(total=file_count, desc="Preprocessing images", unit="image") as pbar:
        # Iterate through the source folder and its sub-folders
        # root = current directory, dirs = subdirectories within current directory,
        # files = filenames in current directory
        for root, dirs, files in os.walk(args.source_folder):
            for file in files:
                try:
                    # Check if the file is an image (any file with an image extension)
                    if any(file.lower().endswith(image_ext) for image_ext in image_extensions):

                        # Build the full path for the source and destination images.
                        # relpath is used to replicate source directory structure.
                        src_image_path = os.path.join(root, file)
                        dest_image_path = os.path.join(args.destination_folder,
                                                       os.path.relpath(src_image_path, args.source_folder))

                        # Create the destination folder if it doesn't exist
                        os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
                        # exist_ok ensures function doesn't raise error if directory already exists

                        # Determine destination file path/name after conversion to jpg
                        dest_image_path = os.path.splitext(dest_image_path)[0] + '.jpg'

                        pbar.set_description(f"Preprocessing image: {os.path.basename(src_image_path)}")

                        # Process images using command-line arguments to meet Transkribus upload requirements
                        meet_upload_reqs(src_image_path, dest_image_path, args.basic_only)

                        if args.basic_only:
                            # Update tqdm progress bar
                            pbar.update(1)
                            continue

                        else:
                            # Pre-process images to prepare them for OCR/HTR (we use destination_folder as the source
                            # folder as images in destination folder have already been partially prepared by
                            # meet_upload_reqs).
                            preprocess_image(dest_image_path,
                                             dest_image_path,
                                             args.contrast_enhance,
                                             args.k_val,
                                             args.window_size)

                            # Update tqdm progress bar
                            pbar.update(1)

                except Exception as image_processing_error:
                    print(f"Error preprocessing image {file}: {image_processing_error}")

    print("Preprocessing completed")
