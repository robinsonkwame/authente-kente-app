from functools import partial
from keras.models import Sequential
#from keras.layers.core import Dense
#from keras.optimizers import SGD
#from keras.utils import to_categorical
from sklearn.metrics import classification_report, roc_auc_score
from pyimagesearch import config
import pandas as pd
import numpy as np
import pickle
import os

# Packages for gridsearch, examining results
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NeighborhoodComponentsAnalysis

def load_data(data_set,
			  base_path=config.BASE_CSV_PATH,
			  remap_y_values={0:-1},
			  use_hsv=False,
			  subset=False):
	hsv = ''
	if use_hsv:
		hsv = 'hsv.'
	data = np.load(
		os.path.sep.join(
			[base_path,
			 f"{data_set}.{hsv}npy"]
		)
	)

	X, y =\
		data[:, config.LABEL_INDEX+1:],\
			data[:, config.LABEL_INDEX]

	if remap_y_values:
		y =\
			np.array(
				[remap_y_values.get(value, value) for value in y]
		)

	if subset:
		X, y = subset_data(X, y)

	return X, y

def subset_data(X, y, fraction=config.PROPORTION_TRAIN_CASES):
	indices = np.random\
				.randint(X.shape[0],
						 size=int(X.shape[0]*fraction))
	return X[indices], y[indices]

def train_pred ():

	# load the label encoder from disk
	le = pickle.loads(open(config.LE_PATH, "rb").read())


	if config.MODEL == "ONECLASS":

		X_train, y_train = load_data(data_set="training.mobile",
								     use_hsv=False,
									 subset=False)  # note one class, need to steal from val

		X_val, y_val = load_data(data_set="validation.mobile",
								 use_hsv=False,
								 subset=False)

		X_test, y_test = load_data(data_set="evaluation.mobile",
								   use_hsv=False,
								   subset=False)

		ss = StandardScaler()
		ss.fit(X_train)
		X_train = ss.transform(X_train)
		X_val = ss.transform(X_val)
		X_test = ss.transform(X_test)
		pickle.dump(ss, open('standardscaler.pkl','wb'))

		#  Here we apply a variety of dimensionality reduction techniques,
		# to be evaluated on held out validation data and within parameter search
		reduction_sizes =\
			{#"reduction_125": int(X_train.shape[0]*0.125),}
			 "reduction_50": int(X_train.shape[0]*0.5)}
			 #"reduction_25": int(X_train.shape[0]*0.25)}
		dimensionality_reducers =\
			{"PCA": PCA,
			 "KPCA": KernelPCA,
			 "NCA": NeighborhoodComponentsAnalysis}

		kpca_args =\
			{"n_jobs": -1,
			 "kernel": "rbf",
			 "copy_X": False}

		fast_ica_args =\
			{"algorithm": "parallel",
			 "max_iter": 400}

		the_reducers = []

		for size in reduction_sizes.values():
			for name, reducer in dimensionality_reducers.items():
				args = {"n_components": size,
						"random_state": 42}
				if name in ["KPCA"]:
					args.update(kpca_args)
				if name in ["NCA"]:
					the_reducers.append(
						(name, reducer(**args).fit(X_train, y_train))
					)
					pass
				else:
					the_reducers.append(
						(name, reducer(**args).fit(X_train))
					)

		#  ... and also do fast ICA with 4 and 2 sources
		ica_reducers =\
			{"FastICA": FastICA}
		for size in [2, 4]:
			for name, reducer in ica_reducers.items():
				args = {"n_components": size,
						"random_state": 42}
				args.update(fast_ica_args)
				the_reducers.append(
					(name, reducer(**args).fit(X_train))  # we could do fit_transform
				)

		the_X_train_embeded = []
		the_X_val_embedded = []

		# plot first two dimensions to get a sense of seperation
		for name, reducer in the_reducers:
			X_embedded = reducer.transform(X_train)
			title_to_plot = name + "_" + str(reducer.n_components)
			# visualize_data(X_embedded[:,:4],
			# 			   y_train,
			# 			   title=title_to_plot)
			the_X_train_embeded.append(X_embedded)

			the_X_val_embedded.append(
				reducer.transform(X_val)
			)
			plt.close("close")  # to prevent too many figs at once

		n_neighbors = {"n_neighbors": [1,5,11]}
		metric = {"metric":\
			['cityblock', 'cosine', 'euclidean',
			 'l1', 'l2', 'manhattan'] +\
			['correlation', 'seuclidean', 'sqeuclidean']}
		novelty = {"novelty":[True]}

		parameter_grid = {**n_neighbors,
						  **metric,
						  **novelty}

		best_clf_with_report = []
		for X_embedded, X_val_embedded, reducer in \
			zip(the_X_train_embeded, the_X_val_embedded, the_reducers):

			folds = StratifiedKFold(n_splits=3).split(X_train, y_train)
			search = GridSearchCV(
				estimator=LocalOutlierFactor(),
				param_grid=parameter_grid,
				scoring=('f1_macro'),
				cv=folds,
				verbose=5,
				n_jobs=-1,
				)

			search.fit(X_embedded,
					   y_train)
			optimal_knn_for_embedding = search.best_estimator_

			preds = optimal_knn_for_embedding.predict(X_val_embedded)
			avg_precision = average_precision_score(y_val, preds)
			report = classification_report(y_val, preds, output_dict=True)
			best_clf_with_report.append(
				(str(reducer),
				 avg_precision,
				 report,
				 optimal_knn_for_embedding)
			)




		n_estimators = {"n_estimators": [50,200,11]}
		n_jobs = {"n_jobs":[-1]}

		parameter_grid = {**n_estimators,
						  **n_jobs}

		for X_embedded, X_val_embedded, reducer in \
			zip(the_X_train_embeded, the_X_val_embedded, the_reducers):

			folds = StratifiedKFold(n_splits=3).split(X_train, y_train)
			search = GridSearchCV(
				estimator=IsolationForest(),
				param_grid=parameter_grid,
				scoring=('f1_macro'),
				cv=folds,
				verbose=5,
				n_jobs=-1,
				)

			search.fit(X_embedded,
					   y_train)
			optimal_forest_for_embedding = search.best_estimator_

			preds = optimal_forest_for_embedding.predict(X_val_embedded)
			avg_precision = average_precision_score(y_val, preds)
			report = classification_report(y_val, preds, output_dict=True)
			best_clf_with_report.append(
				(str(reducer),
				 avg_precision,
				 report,
				 optimal_forest_for_embedding)
			)



		#  ... ^ doesn't do much at all, stick with LOF

		# ... finally try with a simple linear regressor
		for X_embedded, X_val_embedded, reducer in \
			zip(the_X_train_embeded, the_X_val_embedded, the_reducers):
			regressor = LogisticRegressionCV(n_jobs=-1,
											 cv=3,
											 scoring='f1_macro',
											 random_state=42)
			regressor.fit(X_embedded,
						  y_train)
			preds = regressor.predict(X_val_embedded)
			avg_precision = average_precision_score(y_val, preds)
			report = classification_report(y_val, preds, output_dict=True)
			best_clf_with_report.append(
				(str(reducer),
				 avg_precision,
				 report,
				 regressor)
			)


		# ... which is clearly the better overal model here

		#  ... finally we test out of sample to get reportable statistics
		# on the evaluation dataset
		best_classifier_index = -3
		best_reducer_index = 2
		best_reducer_index = 0
		best_classifier =\
			LogisticRegressionCV(
				**best_clf_with_report[best_classifier_index][-1].get_params())
		best_classifier.fit(
			X=np.vstack((the_X_train_embeded[best_reducer_index],
						 the_X_val_embedded[best_reducer_index])),
			y=np.hstack((y_train, y_val))
		)
		X_test_embedded = the_reducers[best_reducer_index][1].transform(X_test)
		preds = best_classifier.predict(X_test_embedded)
		avg_precision = average_precision_score(y_test, preds)
		report = classification_report(y_test, preds)

		# ... this is where we drop our microphone



		#   on PCA, which has nearly similar results as NCA, we get
		best_classifier_index = -3
		best_reducer_index = 0
		best_classifier =\
			LogisticRegressionCV(
				**best_clf_with_report[best_classifier_index][-1].get_params())
		best_classifier.fit(
			X=np.vstack((the_X_train_embeded[best_reducer_index],
						 the_X_val_embedded[best_reducer_index])),
			y=np.hstack((y_train, y_val))
		)
		X_test_embedded = the_reducers[best_reducer_index][1].transform(X_test)
		preds = best_classifier.predict(X_test_embedded)
		avg_precision = average_precision_score(y_test, preds)
		finalreport = classification_report(y_test, preds, output_dict = True)


	pickle.dump(best_classifier, open('model.pkl','wb'))
	pickle.dump(the_reducers[best_reducer_index][1], open('reducer.pkl','wb'))
	return (finalreport)
# train_pred()
