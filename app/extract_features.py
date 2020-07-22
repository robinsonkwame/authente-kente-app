# USAGE
# python extract_features.py

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from pyimagesearch import config
from pathlib import Path
import numpy as np
import pickle
import random
import logging
import os
import pdb
from feature_processor import FeatureProcessor


def create_features(
    base_path,
    folder,
    feature_processor,
    label_encoder,
    feature_file_base_path
):
    # grab all image paths in the current split
    print("[INFO] processing '{} split'...".format(folder))
    p = os.path.sep.join([base_path, folder])

    imagePaths = list(Path(p).glob("*.jpg"))  # for Kente

    print("[INFO] ... number of images in path {} ...".format(len(imagePaths)))
    # randomly shuffle the image paths and then extract the class
    print("[INFO] ... path is {} ...".format(p))

    # labels from the file paths
    random.shuffle(imagePaths)

    #  for Kente we need to do things slightly differently
    labels = [p.name.split("_", 1)[0] for p in imagePaths]

    feature_file_path = Path(
        os.path.sep.join([feature_file_base_path, "{}.{}.{}".format(folder,
        feature_processor.name, feature_processor.feature_file_format)])
    )


    if feature_file_path.exists():
        print(f"[INFO] ... deleting old data for {folder}")
        feature_file_path.unlink()

    feature_processor.initialize_output_processor(labels, feature_file_path)

    for (batch, index) in enumerate(range(0, len(imagePaths), feature_processor.batch_size)):
        # extract the batch of images and labels, then initialize the
        # list of actual images that will be passed through the network
        # for feature extraction
        print(
            "[INFO] processing batch {}/{}".format(
                batch + 1, int(np.ceil(len(imagePaths) / float(feature_processor.batch_size)))
            )
        )
        print("[INFO] label encoding from path ...")
        batchPaths = imagePaths[index : index + feature_processor.batch_size]
        batchLabels = label_encoder.transform(labels[index : index + feature_processor.batch_size])
        batchImages = []
        print("[INFO] ... label encoded!")


        # loop over the images and labels in the current batch
        for image_path in batchPaths:
            # load the input image using the Keras helper utility
            # while ensuring the image is resized to 224x224 pixels
            image = feature_processor.process_image(image_path)
            # add the image to the batch
            batchImages.append(image)

        # pass the images through the network and use the outputs as
        # our actual features, then reshape the features into a
        # flattened volume
        print("[INFO] generating features ... ")
        batchImages = np.vstack(batchImages)

        features = feature_processor.create_features(batchImages)

        print("[INFO] ... generated features")

        feature_processor.output_processor.save_features(features)

        print("[INFO] ... saved features!")

    feature_processor.output_processor.write_to_file()

def extract_features():
    random.seed(1)

    print("[INFO] loading network...")
    batch_size = config.BATCH_SIZE
    feature_file_format = config.FEATURE_FILE_FORMAT

    feature_processor = FeatureProcessor.create(config.FEATURE_PROCESSOR,
                        batch_size, feature_file_format)

    le = LabelEncoder()
    le.fit(["fake", "real"])

    # Create train features
    create_features(
        config.BASE_PATH,
        config.TRAIN,
        feature_processor,
        le,
        config.BASE_CSV_PATH
    )

    # Create test features
    create_features(
        config.BASE_PATH,
        config.TEST,
        feature_processor,
        le,
        config.BASE_CSV_PATH
    )

    # Create validation features
    create_features(
        config.BASE_PATH,
        config.VAL,
        feature_processor,
        le,
        config.BASE_CSV_PATH
    )

    # serialize the label encoder to disk
    f = open(config.LE_PATH, "wb")
    f.write(pickle.dumps(le))
    f.close()
