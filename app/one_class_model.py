# USAGE
# python train.py

# import the necessary packages
import os
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # for visualization
# Packages for gridsearch, examining results
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.metrics import classification_report, average_precision_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns  # for pretty plot
from pyimagesearch import config
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from training_models import LOFTrainer, IsolationForestTrainer, LogisticRegressionTrainer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


sns.set_style("white")


def get_visualization_pipeline():
    pipeline = Pipeline(
        [
            ("standard_scaler", StandardScaler()),
            ("pca", PCA(n_components=3, random_state=42)),
        ]
    )
    return pipeline


def visualize_data(X, y, pred_y=None, title=""):
    my_dpi = 96

    if X.shape[1] >= 3:
        fig = plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)

        #fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlabel("x_composite_3")

        sc = ax.scatter(
            X[:, 0],
            X[:, 1],
            X[:, 2],
            c=y,  # color by outlier or inlier
            cmap="Paired",
            s=20,
            alpha=0.75,
        )

        # Plot x's for the ground truth outliers
        ax.scatter(
            X[y == -1, 0],
            X[y == -1, 1],
            zs=X[y == -1, 2],
            lw=2,
            s=60,
            marker="x",
            c="red",
        )

        labels = np.unique(y)
        print("labels are: ", labels)
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=sc.cmap(sc.norm(yi)))
            for yi in labels
        ]
        plt.legend(handles, labels)

        if pred_y is not None:
            print("[INFO] ... plotting predicted inliers")
            # Plot circles around the predicted outliers
            ax.scatter(
                X[pred_y == -1, 0],
                X[pred_y == -1, 1],
                zs=X[pred_y == -1, 2],
                lw=4,
                marker="o",
                facecolors=None,
                s=80,
            )

        # make simple, bare axis lines through space:
        xAxisLine = ((min(X[:, 0]), max(X[:, 0])), (0, 0), (0, 0))
        ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], "r")
        yAxisLine = ((0, 0), (min(X[:, 1]), max(X[:, 1])), (0, 0))
        ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], "r")
        zAxisLine = ((0, 0), (0, 0), (min(X[:, 2]), max(X[:, 2])))
        ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], "r")

        # label the axes
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"Kente Cloth Inliers and Outliers\n{title}")

        #  plot the figure
        fig.savefig(title + ".png")

        fig.clf()
        plt.close()
        ax.cla()

    # seperately plot the pair wise histogram
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": y})
    if X.shape[1] >= 3:
        df["z"] = X[:, 2]

    grid = sns.pairplot(
        df, hue="label", corner=True, diag_kind="kde"
    )  # todo: set same colors as prior plot
    grid.fig.suptitle(title)
    grid.fig.savefig(title + ".pair" + ".png")
    return


def subset_data(X, y, fraction=config.PROPORTION_TRAIN_CASES):
    indices = np.random.randint(X.shape[0], size=int(X.shape[0] * fraction))
    return X[indices], y[indices]


def load_data(
    data_set,
    base_path=config.BASE_CSV_PATH,
    remap_y_values={0: -1},
    use_hsv=False,
    subset=False,
):
    hsv = ""
    if use_hsv:
        hsv = "hsv."
    data = np.load(os.path.sep.join([base_path, f"{data_set}.{hsv}npy"]))

    X, y = data[:, config.LABEL_INDEX + 1 :], data[:, config.LABEL_INDEX]

    if remap_y_values:
        y = np.array([remap_y_values.get(value, value) for value in y])

    if subset:
        X, y = subset_data(X, y)

    return X, y


def scale_data(X_train, X_val, X_test):
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_val = ss.transform(X_val)
    X_test = ss.transform(X_test)
    pickle.dump(ss, open("standardscaler.pkl", "wb"))
    return (X_train, X_val, X_test)

    #  Here we apply a variety of dimensionality reduction techniques,
    # to be evaluated on held out validation data and within parameter search


def fit_all_reducers(X_train, X_val, X_test, y_train):
    #  Here we apply a variety of dimensionality reduction techniques,
    # to be evaluated on held out validation data and within parameter search

    the_reducers = []
    the_X_train_embeded = []
    the_X_val_embedded = []

    reduction_sizes = {
        "reduction_50": int(X_train.shape[0] * 0.5)
    }

    for size in reduction_sizes.values():
        args = {"n_components": size, "random_state": 42}
        print(f"[INFO] reducing with PCA on size {size} ...")
        reducer = PCA(**args).fit(X_train)
        X_train_embedded = reducer.transform(X_train)
        X_val_embedded = reducer.transform(X_val)
        the_reducers.append(("PCA", reducer))
        the_X_train_embeded.append(X_train_embedded)
        the_X_val_embedded.append(X_val_embedded)

        print(f"[INFO] reducing with NCA on size {size} ...")
        reducer =  NeighborhoodComponentsAnalysis(**args).fit(X_train, y_train)
        X_train_embedded = reducer.transform(X_train)
        X_val_embedded = reducer.transform(X_val)
        the_reducers.append(("NCA", reducer))
        the_X_train_embeded.append(X_train_embedded)
        the_X_val_embedded.append(X_val_embedded)


        print(f"[INFO] reducing with KPCA on size {size} ...")
        kpca_args =\
        {"n_jobs": -1,
         "kernel": "rbf",
         "copy_X": False}
        args.update(kpca_args)
        reducer =  KernelPCA(**args).fit(X_train)
        X_train_embedded = reducer.transform(X_train)
        X_val_embedded = reducer.transform(X_val)
        the_reducers.append(("KPCA", reducer))
        the_X_train_embeded.append(X_train_embedded)
        the_X_val_embedded.append(X_val_embedded)

    #  ... and also do fast ICA with 4 and 2 sources
    for size in [2, 4]:
        print(f"[INFO] ... reducing with FastICA on size {size}")
        args = {"n_components": size, "random_state": 42, "algorithm": "parallel",
         "max_iter": 400}
        reducer =  FastICA(**args).fit(X_train)
        X_train_embedded = reducer.transform(X_train)
        X_val_embedded = reducer.transform(X_val)
        the_reducers.append(("FastICA", reducer))
        the_X_train_embeded.append(X_train_embedded)
        the_X_val_embedded.append(X_val_embedded)

    # # plot first two dimensions to get a sense of seperation
    # for i in range(len(the_reducers)):
    #     name, reducer = the_reducers[i]
    #     X_embedded = the_X_train_embeded[i]
    #     title_to_plot = name + "_" + str(reducer.n_components)
    #     print("[INFO] title to plot is ", title_to_plot)
    #     visualize_data(X_embedded[:, :4], y_train, title=title_to_plot)
    #     plt.close("close")  # to prevent too many figs at once

    return (the_X_train_embeded, the_X_val_embedded, the_reducers)


def train_all_models(
    n_neighbors,
    metric,
    novelty,
    the_X_train_embeded,
    the_X_val_embedded,
    the_reducers,
    X_train,
    y_train,
    y_val
):
    best_clf_with_report_list = []
    for X_embedded, X_val_embedded, reducer in zip(
        the_X_train_embeded, the_X_val_embedded, the_reducers
    ):

        # Try local outlier factor
        trainer = LOFTrainer()
        avg_precision, report, best_estimator = trainer.train(
            X_embedded, X_val_embedded, y_train, y_val
        )
        best_clf_with_report_list.append(
            (str(reducer), avg_precision, report, best_estimator)
        )

        print(str(best_estimator), str(reducer))
        print(
            f"Outlier Recall: {report['-1.0']['recall']:.2f}, Outlier Precision: {report['-1.0']['precision']:.2f}"
        )
        print("avg precision", avg_precision)

        # Try isolation forests
        trainer = IsolationForestTrainer()
        avg_precision, report, best_estimator = trainer.train(
            X_embedded, X_val_embedded, y_train, y_val
        )

        best_clf_with_report_list.append(
            (str(reducer), avg_precision, report, best_estimator)
        )

        print(str(best_estimator), str(reducer))
        print(
            f"Outlier Recall: {report['-1.0']['recall']:.2f}, Outlier Precision: {report['-1.0']['precision']:.2f}"
        )
        print("avg precision", avg_precision)

        # ... finally try with a simple linear regressor
        trainer = LogisticRegressionTrainer()
        avg_precision, report, regressor = trainer.train(
            X_embedded, X_val_embedded, y_train, y_val
        )

        best_clf_with_report_list.append(
            (str(reducer), avg_precision, report, regressor)
        )

        print(str(regressor), str(reducer))
        print(
            f"Outlier Recall: {report['-1.0']['recall']:.2f}, Outlier Precision: {report['-1.0']['precision']:.2f}"
        )
        print("avg precision", avg_precision)

    return best_clf_with_report_list

def run_one_class_model():
    print("[INFO] Loading train, validation and test into memory ...")
    # get all of train, evaluation generator
    # Note we use two class (insead of one class), for NCA
    #  Some of the outliers got moved to validation so, the validation
    # set is slightly baised. The test set should have only outliers not seen before
    #
    # So we do hyperparameter searching, still, on the training, validation set
    # with validation hold out to double check. Then we'll look at test wiht the best one
    # hopefullyit's all good.
    X_train, y_train = load_data(
        data_set="training.mobile", use_hsv=False, subset=False
    )  # note one class, need to steal from val

    X_val, y_val = load_data(data_set="validation.mobile", use_hsv=False, subset=False)

    X_test, y_test = load_data(data_set="evaluation.mobile", use_hsv=False, subset=False)
    print("[INFO] ... loaded into memory")

    print("[INFO] Applying standard scaling ...")


    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

    print("[INFO] ... scaled")


    the_X_train_embeded, the_X_val_embedded, the_reducers = fit_all_reducers(
        X_train, X_val, X_test, y_train
    )


    print("[INFO] Entering hyper parameter search ...")

    n_neighbors = {"n_neighbors": [1, 5, 11]}
    metric = {
        "metric": ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"]
        + ["correlation", "seuclidean", "sqeuclidean"]
    }
    novelty = {"novelty": [True]}

    best_clf_with_report_list = train_all_models(
        n_neighbors,
        metric,
        novelty,
        the_X_train_embeded,
        the_X_val_embedded,
        the_reducers,
        X_train,
        y_train,
        y_val
    )

    for i in best_clf_with_report_list:
        print (f"avg_ precision: {i[1]}", f"Regressor: {i[3]}")

    #  ... finally we test out of sample to get reportable statistics
    # on the evaluation dataset
    best_classifier_index = 2
    best_reducer_index = 0

    print(f"best params: {best_clf_with_report_list[best_classifier_index][-1].get_params()}")

    print("[INFO] ... Training best classifer on validation, training")
    best_classifier = LogisticRegressionCV(
        **best_clf_with_report_list[best_classifier_index][-1].get_params()
    )
    best_classifier.fit(
        X=np.vstack(
            (
                the_X_train_embeded[best_reducer_index],
                the_X_val_embedded[best_reducer_index],
            )
        ),
        y=np.hstack((y_train, y_val)),
    )
    print("[INFO] ... Test trained classifer on hold out evaluation sample")
    X_test_embedded = the_reducers[best_reducer_index][1].transform(X_test)
    preds = best_classifier.predict(X_test_embedded)
    avg_precision = average_precision_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict = True)
    print(report)
    # ... this is where we drop our microphone


    #   on PCA, which has nearly similar results as NCA, we get
    best_classifier_index = 2
    best_reducer_index = 0
    print("[INFO] ... Training best classifer on validation, training")
    best_classifier = LogisticRegressionCV(
        **best_clf_with_report_list[best_classifier_index][-1].get_params()
    )
    best_classifier.fit(
        X=np.vstack(
            (
                the_X_train_embeded[best_reducer_index],
                the_X_val_embedded[best_reducer_index],
            )
        ),
        y=np.hstack((y_train, y_val)),
    )
    print("[INFO] ... Test trained classifer on hold out evaluation sample")
    X_test_embedded = the_reducers[best_reducer_index][1].transform(X_test)
    preds = best_classifier.predict(X_test_embedded)
    avg_precision = average_precision_score(y_test, preds)
    target_names = ['fake', 'real']
    report = classification_report(y_test, preds,
                target_names=target_names, output_dict=True)


    pickle.dump(best_classifier, open("model.pkl", "wb"))
    pickle.dump(the_reducers[best_reducer_index][1], open("reducer.pkl", "wb"))

    return report
