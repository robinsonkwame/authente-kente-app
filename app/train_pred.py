# USAGE
# python train.py

from pyimagesearch import config

def train_pred():
    if config.MODEL == "ONECLASS":
        print("Running one class model")
        #see: https://hackernoon.com/one-class-classification-for-images-with-deep-features-be890c43455d
        #see: https://sdsawtelle.github.io/blog/output/week9-anomaly-andrew-ng-machine-learning-with-python.htmlfrom sklearn.model_selection import StratifiedKFold
        from one_class_model import run_one_class_model
        report = run_one_class_model()
        return report

    elif config.MODEL == 'SGD':
        from sgd_model import run_sgd_model
        run_sgd_model()
