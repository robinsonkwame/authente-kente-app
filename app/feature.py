BATCH_SIZE = 32

def image_pros():
  from tensorflow.keras.applications import MobileNetV2
  from keras.applications import imagenet_utils
  from keras.preprocessing.image import img_to_array
  from keras.preprocessing.image import load_img
  from pathlib import Path
  import numpy as np
  import cv2
  import pickle
  import logging
  import os
  import pickle
  import os
  from pickle import load

# creating features
  logger = logging.getLogger(__name__)
  mobile_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
  mobilenet_flattened_size = 7 * 7 * 1280
  #loading images from the folder
  load_files = os.listdir('.')
  imagePaths = ''
  for file in load_files:
      if file.endswith('jpg') or file.endswith('png'):
          imagePaths = file

  image = load_img(imagePaths, target_size=(224, 224))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = imagenet_utils.preprocess_input(image)
  batchImages = []
  batchImages.append(image)
  batchImages = np.vstack(batchImages)
  features = mobile_model.predict(batchImages, BATCH_SIZE)
  features = features.reshape((features.shape[0], mobilenet_flattened_size))

  # reducing features
  ss = pickle.load( open( "standardscaler.pkl", "rb" ) )
  data = ss.transform(features)
  reducer = pickle.load( open( "reducer.pkl", "rb" ) )
  data = reducer.transform(data)

  # predict
  model = pickle.load( open( "model.pkl", "rb" ) )
  ans = model.predict(data)
  if ans == 1:
      pred = 'real'
  else:
      pred = 'fake'
  return(pred)
