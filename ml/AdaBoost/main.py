
"""
Info    Implementation of AdaBoost, basic classifier is Decision Stump.
Author  Yiqun Chen
Time    2020-04-18
"""

from ml.utils.metrics import Logger
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

logger = Logger()

def load_data(dataset="iris"):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    data[:, -1][data[:, -1] < 0.5] = -1
    return data[:,:4], data[:,-1].astype(np.int8)

class AdaBoost(object):
    """
    Info    AdaBoost.
    Args    M (int): how many basic classifier the model need. Default is 1 for the small dataset.
            dim (int): which dimension of the data the model will use to train and predict. 
            classifier_type (string): which type the basic classifier is, now only "DecisionStump" available.
    Returns AdaBoost model.
    """
    def __init__(self, M=1, dim=0, classifier_type="DecisionStump"):
        self.M = M
        self.dim = dim
        self.alpha = np.ones(M)
        self.classifiers = []
        self.classifier_type = classifier_type
        # self._build_model()

    # def _build_model(self, dim):
    #     weight = np.ones(dim) * 1.0 / dim
    #     for m in range(self.M):
    #         self.classifiers.append(eval(self.classifier_type)(self.dim, weight))

    def __call__(self, x):
        """
        Info    predict the label of the input data.
        Args    x (ndarray): data to be predicted. 
        Returns the predict label of the input data.
        """
        return self.predict(x)

    def train(self, x, y):
        """
        Info    train the adaboost model, the model will record the trained basic classifiers.
        Args    x (ndarray): the training data.
                y (ndarray): the label of the training data. 
        Returns None.
        """
        if len(x.shape) > 1:
            x = x[:, self.dim]
        weight = np.ones(y.shape[0]) * 1.0 / y.shape[0]
        self.classifiers = []
        self.error_rate = []
        pred = np.zeros(y.shape[0])
        for m in range(self.M):
            classifier = eval(self.classifier_type)(weight)
            classifier.train(x, y)
            # pred = classifier(x)
            error_rate = classifier.error_rate
            self.error_rate.append(error_rate)
            self.classifiers.append(classifier)
            pred = np.zeros(y.shape[0])
            for c in self.classifiers:
                pred += c(x)
            pred[pred >= 0] = 1
            pred[pred < 0] = -1
            x = (y - pred).astype(np.float32)
            # self.alpha[m] = 0.5 * np.log((1-error_rate)/error_rate)
            # weight = weight * np.exp(-self.alpha[m] * pred * y)
            # weight = weight / np.sum(weight)

    def predict(self, x):
        """
        Info    predict the label of the input data.
        Args    x (ndarray): input data. 
        Returns y (ndarray): predicted label of the input data.
        """
        if len(x.shape) > 1:
            x = x[:, self.dim]
        if len(self.classifiers) == 0:
            self._build_model(x.shape[self.dim])
        pred = np.zeros(x.shape[0])
        for i in range(len(self.classifiers)):
            pred += self.classifiers[i](x) * self.alpha[i]
        y = pred > 0
        y = y.astype(np.int8)
        y[y==0] = -1
        return y


class DecisionStump(object):
    """
    Info    Decision Stump class.
    """
    def __init__(self, weight=None):
        self.threshold = 0.0
        # self.dim = 0
        self.weight = weight

    def __call__(self, x):
        """
        Info    predict the label of input.
        Args    x (ndarray): input data.
        Returns y (ndarray): labels predicted.
        """
        return self.predict(x)

    def train(self, x, y):
        """
        Info    train the DecisionStump model.
        Args    x (ndarray): 1-D training data.
                y (ndarray): label of the training data.
        Returns None.
        """
        if self.weight is None:
            self.weight = np.ones(x.shape[0]) * 1.0/x.shape[0]
        pred = self.predict(x)
        x_max, x_min = np.max(x), np.min(x)
        param = []
        error_rate = []
        thresholds = np.arange(x_min-0.05, x_max+0.15, 0.1)
        for threshold in thresholds:
            self.threshold = threshold
            pred = self.predict(x)
            error_rate.append(np.sum(self.weight * (pred != y).astype(np.float32)))
        # self.error_rate = np.sum(self.weight * (pred == y).astype(np.float32))
        self.error_rate = min(error_rate)
        self.threshold = x_min - 0.05 + 0.1 * error_rate.index(self.error_rate)

    def predict(self, x):
        """
        Info    predict the label of input.
        Args    x (ndarray): input data.
        Returns y (ndarray): labels predicted.
        """
        if self.weight is None:
            self.weight = np.ones(x.shape[0]) * 1.0/x.shape[0]
        y = x > self.threshold
        y = y.astype(np.int8)
        y[y==0] = -1
        return y


if __name__ == "__main__":
    logger.record_item_time("read_data")
    logger.log_info("reading data...")
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=65)
    logger.log_info("done after {:.3f} seconds".format(logger.show_item_time("read_data")))

    model = AdaBoost(M=2, dim=2)

    logger.log_info("training start!")
    logger.record_item_time("train")
    model.train(X_train, Y_train)
    logger.log_info("training finished after {:.3f} seconds".format(logger.show_item_time("train")))

    pred = model(X_test)
    accuracy = accuracy_score(Y_test, pred)
    logger.log_info("accuracy: {:.3f}".format(accuracy))
    logger.log_info("reach the end of program.")