def extract_features ():
	# import the necessary packages
	from sklearn.preprocessing import LabelEncoder
	from keras.applications import ResNet50, MobileNetV2
	from keras.applications import imagenet_utils
	from keras.preprocessing.image import img_to_array
	from keras.preprocessing.image import load_img
	from pyimagesearch import config
	from pathlib import Path
	import numpy as np
	import cv2
	import pickle
	import random
	import logging
	import os
	import pdb

	logger = logging.getLogger(__name__)
	random.seed(1)

	# load the ResNet50 network and initialize the label encoder
	mobile_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
	mobilenet_flattened_size = 7 * 7 * 1280
	flattened_size = 7 * 7 * 2048

	le = None

	# loop over the data splits
	for split in (config.TRAIN, config.TEST, config.VAL):
	# grab all image paths in the current split
		p = os.path.sep.join([config.BASE_PATH, split])
		#imagePaths = list(paths.list_images(p))
		#imagePaths = list(Path(p).glob("*/*.jpg")) # for 5k
		imagePaths = list(Path(p).glob("*.jpg")) # for Kente

		# randomly shuffle the image paths and then extract the class

		# labels from the file paths
		random.shuffle(imagePaths)
		#labels = [str(p).split(os.path.sep)[-2] for p in imagePaths]
		#  for Kente we need to do things slightly differently
		labels = [p.name.split('_',1)[0] for p in imagePaths]

	# if the label encoder is None, create it
		if le is None:
			le = LabelEncoder()
			# le.fit(labels)
			#  the above assumes all label types are present but
			# in training they aren't
			le.fit(['fake','real'])

	# open the output CSV file for writing

		npPath = Path(
			os.path.sep.join([config.BASE_CSV_PATH,
			"{}.npy".format(split)])
			)

		mobilenetPath = Path(
			os.path.sep.join([config.BASE_CSV_PATH,
			"{}.mobile.npy".format(split)])
			)

		if config.EXTRACT_FEATURES_TO_NPY:
			if npPath.exists():
				npPath.unlink()
			feature_array = np.zeros(
				(
					len(labels),
					flattened_size + 1
				)  # +1 for labels
			)

		if config.EXTRACT_MOBILENET_FEATURES:
			if mobilenetPath.exists():
				npPath.unlink()
			feature_array = np.zeros(
				(
					len(labels),
					mobilenet_flattened_size + 1
				)  # +1 for labels
			)

		# loop over the images in batches
		for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
			# extract the batch of images and labels, then initialize the
			# list of actual images that will be passed through the network
			# for feature extraction
			batchPaths = imagePaths[i: i + config.BATCH_SIZE]
			batchLabels = le.transform(labels[i: i + config.BATCH_SIZE])
			batchImages = []

		# loop over the images and labels in the current batch
			for imagePath in batchPaths:
				# load the input image using the Keras helper utility
				# while ensuring the image is resized to 224x224 pixels
				image = load_img(imagePath, target_size=(224, 224))

				if config.EXTRACT_MOBILENET_FEATURES:
					image = img_to_array(image)
					# preprocess the image by (1) expanding the dimensions and
					# (2) subtracting the mean RGB pixel intensity from the
					# ImageNet dataset
					image = np.expand_dims(image, axis=0)
					image = imagenet_utils.preprocess_input(image)
					# add the image to the batch
					batchImages.append(image)

			batchImages = np.vstack(batchImages)

			if config.EXTRACT_MOBILENET_FEATURES:
				features = mobile_model.predict(batchImages, batch_size=config.BATCH_SIZE)
				features = features.reshape((features.shape[0], mobilenet_flattened_size))


			if config.EXTRACT_FEATURES_TO_NPY or config.EXTRACT_AS_HSV\
				or config.EXTRACT_MOBILENET_FEATURES:
				# add vector to feature file for faster read
				labels_by_column =\
					np.array(
						[0 if label == 'fake' else 1 for
							label in labels[i: i + config.BATCH_SIZE]],
						ndmin= 2
					)

				feature_array[
					i : i + config.BATCH_SIZE,
					: ] =\
						np.concatenate(
							(labels_by_column.T, features),
							axis=1
						)

		# write out the full feature array
		if config.EXTRACT_FEATURES_TO_NPY:
			np.save(npPath, feature_array)

		# write out the full feature array
		if config.EXTRACT_MOBILENET_FEATURES:
			np.save(mobilenetPath, feature_array)


	# serialize the label encoder to disk
	f = open(config.LE_PATH, "wb")
	f.write(pickle.dumps(le))
	f.close()

# extract_features()
