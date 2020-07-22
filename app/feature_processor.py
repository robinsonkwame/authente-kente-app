from abc import ABC, abstractmethod
import numpy as np
from keras.applications import ResNet50, MobileNetV2
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

class FeatureProcessor(ABC):
	def __init__(self,
				batch_size,
				flattened_size,
				feature_file_format
				):
		super().__init__()
		self.batch_size = batch_size
		self.flattened_size = flattened_size
		self.feature_file_format = feature_file_format

	@staticmethod
	def create(feature_processor_name, batch_size, feature_file_format):
		if feature_processor_name == "MobileNet":
			flattened_size = 7 * 7 * 1280
			return MobileNetFeatureProcessor(batch_size, flattened_size, feature_file_format)
		elif feature_processor_name == "ImageNet":
			flattened_size = 7 * 7 * 2048
			return ImageNetFeatureProcessor(batch_size, flattened_size, feature_file_format)
		elif feature_processor_name == "HSV":
			flattened_size = 224 * 224 * 3
			return HsvFeatureProcessor(batch_size, flattened_size, feature_file_format)

	def initialize_output_processor(self, labels, feature_file_path):
		if self.feature_file_format == "npy":
			self.output_processor = NpyOutput(labels,
			self.flattened_size, self.batch_size, feature_file_path)
		elif self.feature_file_format == "csv":
			self.output_processor = CsvOutput(labels,
			self.batch_size, feature_file_path)

	@abstractmethod
	def process_image(self):
		pass

	@abstractmethod
	def create_features(self):
		pass

class MobileNetFeatureProcessor(FeatureProcessor):
	def __init__(self, batch_size, flattened_size, feature_file_format):
		super().__init__(batch_size, flattened_size, feature_file_format)
		self.model = MobileNetV2(weights="imagenet",
					include_top=False, input_shape=(224, 224, 3))
		self.name = "mobile"

	def process_image(self, image_path):
		image = load_img(image_path, target_size=(224, 224))
		image = img_to_array(image)
		# preprocess the image by (1) expanding the dimensions and
		# (2) subtracting the mean RGB pixel intensity from the
		# ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)
		return image


	def create_features(self, batch_images):
		features = self.model.predict(batch_images, batch_size= self.batch_size)
		features = features.reshape((features.shape[0], self.flattened_size))
		return features

class ImageNetFeatureProcessor(MobileNetFeatureProcessor):
	def __init__(self, batch_size, flattened_size, feature_file_format):
		super().__init__(batch_size, flattened_size, feature_file_format)
		self.model = ResNet50(weights="imagenet", include_top=False)
		self.name = "image_net"

class HsvFeatureProcessor(FeatureProcessor):
	def __init__(self, batch_size, flattened_size, feature_file_format):
		super().__init__(batch_size, flattened_size, feature_file_format)
		self.name = "hsv"

	def process_image(self, image_path):
		image = load_img(image_path, target_size=(224, 224))
		image = img_to_array(image.convert('HSV'))
		image = np.expand_dims(image, axis=0)
		return image

	def create_features(self, batch_images):
		features = batchImages.reshape((batchImages.shape[0], self.flattened_size))
		return features



class NpyOutput():
	def __init__(self, labels, flattened_size, batch_size, feature_file_path):
		self.labels = labels
		self.feature_array = np.zeros((len(labels), flattened_size + 1))  # +1 for labels
		self.batch_size = batch_size
		self.feature_file_path = feature_file_path
		self.current_index = 0

	def save_features(self, features):
		print("[INFO] extracting features to NPY ...")
		labels_by_column = np.array(
			[0 if label == "fake" else 1 for
				label in self.labels[self.current_index :
				self.current_index + self.batch_size]],
			ndmin=2,
		)

		self.feature_array[self.current_index :
			self.current_index + self.batch_size, :] = np.concatenate(
			(labels_by_column.T, features), axis=1)
		self.current_index += self.batch_size

	def write_to_file(self):
		np.save(self.feature_file_path, self.feature_array)

class CsvOutput():
	def __init__(self, labels, batch_size, feature_file_path):
		super().__init__()
		self.labels = labels
		self.batch_size = batch_size
		self.csv = open(feature_file_path, "w")
		self.current_index = 0

	def save_features(self, features):
		batch_labels = label_encoder.transform(
					self.labels[self.current_index :
					self.current_index + self.batch_size])

		for (label, vec) in zip (batch_labels, features):
			# construct a row that exists of the class label and
			# extracted features
			vec = ",".join([str(v) for v in vec])
			self.csv.write("{},{}\n".format(label, vec))
		self.current_index += self.batch_size

	def write_to_file(self):
		self.csv.close()
