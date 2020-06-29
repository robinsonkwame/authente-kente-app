import random
import cv2
from image_transform import ImageTransformer, save_image, load_image
import fnmatch
import os
import re
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)

# https://www.freecodecamp.org/news/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f/

# change brightness

def change_brightness(image, min_alpha=0.95):
    # note we random.seed from generate_subsections should force
    # determinism

    lightness_scaler = random.uniform(min_alpha, 1)

    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype = np.float64)
    random_brightness_coefficient =\
        random.uniform(lightness_scaler, 1.1)
    image_HLS[:,:,1] = image_HLS[:,:,1] * random_brightness_coefficient
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    return image_RGB

# add a shadow

def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(random.randint(3,15)):
            vertex.append(
                (imshape[1] * random.uniform(0,1),
                 imshape[0]//3 + imshape[0] * random.uniform(0,1))
            )
            vertices = np.array([vertex], dtype=np.int32)
            vertices_list.append(vertices)
    return vertices_list

def add_shadow(image, no_of_shadows=1, min_alpha=0.95):
    # note we random.seed from generate_subsections should force
    # determinism
    lightness_scaler = random.uniform(min_alpha, 1)

    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list = generate_shadow_coordinates(imshape, no_of_shadows)
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 10)
        # note: L in HLS is last dimension, not second as suggested
        # by the name HLS
        image_HLS[:,:,1][mask[:,:,0]==10] =\
            image_HLS[:,:,1][mask[:,:,0]==10] * lightness_scaler
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    return image_RGB

def find_files(pattern, directory='.'):
    """
    Finds the images by pattern and returns the list

    Given a pattern and directory, goes through the directory and returns
    the images based on the given pattern. Igores case.

    Parameters
    ----------
    pattern : str
        Regex pattern for desired image
    directory : str
       Directory path containing the images

    Returns
    -------
    arr
        Array of names that match the given pattern.

    """
    rule = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    return [name for name in os.listdir(directory) if rule.match(name)]

def generate_subsections(seed,
                         N,
                         W,
                         H,
                         input_filepath,
                         output_filepath,
                         target_width=None,
                         target_height=None,
                         xyz = [0,0,0],
                         min_alpha=0.95):
    """
    Usage
    ----------
    Change main function with desired arguments
    Then
    from process_image import generate_subsections

    Processes desired directory of images to create random subsections.

    Parameters
    ----------
    seed              : Seed value that allows this process to be deterministic
    N                 : Number of sub image/sub sections desired
    W                 : Width of subsection
    H                 : Height of
    input_filepath    : Location of images to be processed
    output_filepath   : Location of subsections to be saved
    target_width      : Downsampled width
    target_height     : Downsampled height
    xyz               : Tuple containing desired rotation.
                    Default is (0,0,0)

    Output
    ----------
    subsections       : Random subsections of each image

    Note: For the rotation, opencv's warpperspective was used. The bordertype is set
    to reflect instead of having a background fill. Also, in ImageTransformer, you
    are able to set an offset for each image.
    """

    #  Contains random seed that allows this process to be deterministic
    random.seed(seed)

    # To contain the list of images in given directory
    image_list = []

    # Creates new ImageTransformer for each image in directory.
    # Appends tuple of ImageTranformer and image name for later use
    for filename in find_files('*.jpg', directory=input_filepath):
        img = ImageTransformer(input_filepath + filename, None)
        img_name = filename.rsplit('.', 1)
        image_list.append((img, img_name[0]))

    # For each image, generate N number of subsections, randomly located on
    # the image. Each subsection is has a width of W, and a height of H.
    # Each image will be rotated first, before cropping.
    # Output will be saved to given file path.
    # NameOfImage_NumberOfSubsection.jpg
    for img in image_list:
        logger.info('\t... processing {}'.format(img[1]))
        for x in range(0, N):
            # Offset size of subsection to avoid grabbing incomplete image.
            left = random.randint(0, img[0].width-W)
            upper = random.randint(0, img[0].height-H)
            right = left + W
            lower = upper + H

            #Rotate image given x,y,z
            rotated_img = img[0].rotate_along_axis(theta = xyz[0],
                                                   phi=xyz[1],
                                                   gamma=xyz[2])

            # Crop image based of subsection dimensions
            crop_image = rotated_img[upper:lower, left:right]

            # Change brightness
            brightness = change_brightness(crop_image, min_alpha=min_alpha)

            # Add a shadow
            save_img = add_shadow(brightness, no_of_shadows=1, min_alpha=min_alpha)

            # Resize if target_{width, height} provided
            if target_height and target_width:
                save_img =\
                    img[0].downsample(
                        image=save_img,
                        width=target_width,
                        height=target_height)

            # Save image to output filepath
            save_image(
                output_filepath + img[1] + '_%d.jpg' % (x),
                save_img)


#Example
# generate_subsections(3123412,12,300,300,"../../data/raw/","../../data/processed/",(0,0,0))
