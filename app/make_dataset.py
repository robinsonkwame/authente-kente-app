# -*- coding: utf-8 -*-
import click
import logging
import cv2
import os
import shutil
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from process_image import generate_subsections
from dotenv import find_dotenv, load_dotenv
from itertools import chain
import pandas as pd
import random

input_filepath = 'raw'
seed = 0
width = 224
height = 224
target_width = 224
target_height = 224
xrotation = 40
yrotation = 40
zrotation = 10
create_balanced_real_fake = 1000
interim_directory = "interim/"
whole_cloth_stratify = [-1, -1, 4]
fake_prefix = 'fake'
real_prefix = 'real'

def load_image(img_path, shape=None):
    img = cv2.imread(img_path, flags=1)
    if shape is not None:
        img = cv2.resize(img, shape)

    return img

def get_inlier_outlier_groups(num_samples, inlier_groups, outlier_groups):
    """
    Get even number of unique inlier and outlier groups
    """
    num_inliers = int(num_samples / 2)
    num_outliers = num_samples - num_inliers
    inlier_groups = random.sample(inlier_groups, num_inliers)
    outlier_groups = random.sample(outlier_groups, num_outliers)
    groups = inlier_groups + outlier_groups
    return groups

def makeinterim():
    # NOTE: Could make a lot more DRY but this is clearer and this is a
    # single use util function
    #  make fake, real subdirectories within iterim so we can
    # apply differetn parametrizations generate subsections w/o thinking
    fake_interim_directory = Path("interim/fake/")
    fake_interim_directory.mkdir(parents=True, exist_ok=True)
    real_interim_directory = Path("interim/real/")
    real_interim_directory.mkdir(parents=True, exist_ok=True)

    # Creating number_per_fake and number_per_real

    real = 0
    fake = 0

    for filename in os.listdir('raw'):

        if filename.startswith('fake'):
            fake = fake + 1

        elif filename.startswith('real'):
            real = real + 1

    number_per_real = int(create_balanced_real_fake / real)
    number_per_fake = int(create_balanced_real_fake / fake)

    if not Path(input_filepath).exists():
        return
        # raise FileNotFoundError, "Input file path does not exist!"

    # Then we copy each type of image into its respective directory ...
    for prefix, directory in [(fake_prefix + '*', fake_interim_directory),
                              (real_prefix + '*', real_interim_directory)]:
        for image in Path(input_filepath).glob(prefix):
            shutil.copy(image, str(directory))

    #  ... generate subsections against both
    for the_source_path, number_of_images in [(str(fake_interim_directory), number_per_fake),
                                              (str(real_interim_directory), number_per_real)]:
        generate_subsections(seed,
                             number_of_images,
                             width,
                             height,
                             the_source_path + '/',
                             interim_directory,
                             target_height,
                             target_width,
                             (xrotation, yrotation, zrotation))

    # ... finally, we clean up the interim directories
    for directory in [fake_interim_directory, real_interim_directory]:
        for the_file in directory.glob('*'):
            the_file.unlink()
        directory.rmdir()



def makeprocessed():
    #  This function copies out interim images into training, evaluation and validation
    # training datasets into processed.
    inlier = 1
    outlier = -1
    interim_directory = 'interim'
    evaluation_directory = 'processed/evaluation'
    training_directory = 'processed/training'
    validation_directory = 'processed/validation'
    interim_directory = Path(interim_directory)
    evaluation_directory = Path(evaluation_directory)
    evaluation_directory.mkdir(exist_ok=True)

    training_directory = Path(training_directory)
    training_directory.mkdir(exist_ok=True)

    validation_directory = Path(validation_directory)
    validation_directory.mkdir(exist_ok=True)

    number_of_images = len(list(interim_directory.glob('*.jpg')))
    y = np.full((number_of_images,), inlier)
    the_groups = []
    the_file_paths = []
    for index, file_name in enumerate(interim_directory.glob('*.jpg')):
        label = file_name.name.split('_')[0]
        group = file_name.name.rsplit('_',1)[0]
        if label == 'fake':
            y[index] = outlier
        the_groups.append(group)
        the_file_paths.append(file_name)

    sampling_frame = \
        pd.DataFrame(
            {"group": the_groups,
             "file_path": the_file_paths,
             "y": y}
        )

    if len(whole_cloth_stratify) == 0:
        #  ... we split this up into evaluation (testing) and then the remaining we
        # split into validation and training. We focus top down on how
        # many inlier and outlier groups the evaluation set should have, randomly chose those
        # then split the remaining instances 50%/50% to create the validation, training
        # groups.
        #
        # This assumes that the evaluation groups represent a balanced set; the remaining
        # data is stratified so that's much more balanced by design. If the original
        # data is balanced (which is is in my case) then everything will be roughly balanced
        inlier_test_groups = \
            set(
                sampling_frame.sample(frac=1, random_state=42) \
                    .query(f'y=={inlier}') \
                    .group \
                    .unique()[:number_inlier_test_groups]
            )

        outlier_test_groups = \
            set(
                sampling_frame.sample(frac=1, random_state=42) \
                    .query(f'y=={outlier}') \
                    .group \
                    .unique()[:number_outlier_test_groups]
            )
        test_groups = inlier_test_groups | outlier_test_groups

        # ... finally we split the remaining data into balanced training and validation
        validation_training_df = \
            sampling_frame.query('group not in @test_groups')
        validation_indices, training_indices = \
            next(
                StratifiedShuffleSplit(random_state=42,
                                       n_splits=1,
                                       test_size=0.5).split(
                    validation_training_df.drop('y', axis=1),
                    validation_training_df.y)
            )

    else:
        num_train, num_validation, num_eval = whole_cloth_stratify
        random.seed(6)

        inlier_groups = list(sampling_frame.query(f'y=={inlier}').group.unique())
        outlier_groups = list(sampling_frame.query(f'y=={outlier}').group.unique())
        num_unique = len(sampling_frame.group.unique())

        if num_eval + num_train + num_validation > num_unique:

            raise ValueError("whole_cloth_stratify: sum of arguments is greater than number of unique images")

        #If num_eval is -1 then use 1/3rd of unique images
        if num_eval < 0:
            num_eval = round(num_unique / 3)

        #
        test_groups = get_inlier_outlier_groups(num_eval, inlier_groups, outlier_groups)
        inlier_groups = [i for i in inlier_groups if i not in test_groups]
        outlier_groups = [i for i in outlier_groups if i not in test_groups]

        validation_training_df = sampling_frame.query('group not in @test_groups').reset_index(drop=True)

        if num_train > 0 and num_validation > 0:
            # Randomly pick num_train unique images for training set
            # then randomly pick num_validation unique images for validation set
            train_groups = get_inlier_outlier_groups(num_train, inlier_groups, outlier_groups)
            inlier_groucps = [i for i in inlier_groups if i not in train_groups]
            outlier_groups = [i for i in outlier_groups if i not in train_groups]

            training_indices = validation_training_df.query('group in @train_groups').index

            validation_groups = get_inlier_outlier_groups(num_validation, inlier_groups, outlier_groups)
            inlier_groups = [i for i in inlier_groups if i not in validation_groups]
            outlier_groups = [i for i in outlier_groups if i not in validation_groups]
            validation_indices = validation_training_df.query('group in @validation_groups').index

        elif num_train < 0 and num_validation < 0:
            # StratifiedShuffleSplit remaining images into training and validation
            validation_indices, training_indices = \
                next(
                    StratifiedShuffleSplit(random_state=42,
                                           n_splits=1,
                                           test_size=0.5).split(
                        validation_training_df.drop('y', axis=1),
                        validation_training_df.y)
                    )
        else:
            raise ValueError("whole_cloth_stratify: invalid values! Should be -1 or a positive integer")

    # Copy images from interim into training, validation and test data
    # folders based on the whole_cloth_stratify

    copy_to_directory = validation_directory
    validation_training_df.iloc[validation_indices] \
        .apply(lambda row: shutil.copy(
        str(row.file_path),
        str(copy_to_directory / row.file_path.name)),
               axis=1)

    copy_to_directory = training_directory
    validation_training_df.iloc[training_indices] \
        .apply(lambda row: shutil.copy(
        str(row.file_path),
        str(copy_to_directory / row.file_path.name)),
               axis=1)

    copy_to_directory = evaluation_directory
    sampling_frame.query('group in @test_groups') \
        .apply(lambda row: shutil.copy(
        str(row.file_path),
        str(copy_to_directory / row.file_path.name)),
               axis=1)
