
"""
Info    Support Vector Machine for course Machine Learning.
Author  Yiqun Chen
Time    2020-04-02
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import copy
import os
from tqdm import tqdm
from ml.utils.metrics import Logger
from sklearn.svm import LinearSVC

class SVM(object):
    """
    Info    Support Vector Machine.
    """

    def __init__(self, kernel="linear", max_iter=5000, epsilon=0.0001, penality_coeff=1):
        super(SVM, self).__init__()
        self._KERNELS = {
            "linear": self._linear_transform, 
            "gaussian": self._gaussian_transform, 
            "polynomial": self._polynomial_transform, 
        }
        assert kernel in self._KERNELS.keys(), "unsupported kernel {}".format(kernel)
        self.kernel = kernel
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.penality_coeff = penality_coeff

    def _setup(self, features, labels):
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.bias = 0.0
        self.alpha = np.zeros(self.features.shape[0])
        ##  TODO calculate E
        self.E = self._cal_E(self.features, self.labels)
        assert self.E.shape == self.labels.shape, "E shape error"
    
    def _gaussian_transform(self, x_1, X_2):
        Exception(">>>> gaussian kernel is not defined yet.")

    def _linear_transform(self, x_1, x_2):
        return np.matmul(x_1, x_2)

    def _polynomial_transform(self, x_1, x_2, p=3):
        return (np.matmul(x_1, x_2) + 1) ** p

    def transform(self, x_1, x_2):
        return self._KERNELS[self.kernel](x_1, x_2)

    def _cal_g(self, x):
        return np.matmul(
            self.alpha * self.labels, self.transform(x, self.features.transpose(-1, -2))
        ) + self.bias

    def _cal_E(self, x, y):
        return self._cal_g(x) - y

    def _satify_stop_condition(self):
        return self._satify_KKT_condition()

    def _satify_KKT_condition(self):
        cond = self.labels * self._cal_g(self.features)
        alpha_is_zero = np.abs(self.alpha) < self.epsilon
        alpha_is_penality_coeff = np.abs(self.alpha - self.penality_coeff) < self.epsilon
        alpha_other = (1 - (alpha_is_zero + alpha_is_penality_coeff)) > 0
        cond_alpha_is_zero = cond[alpha_is_zero] - 1 >= self.epsilon
        cond_alpha_is_penality_coeff = cond[alpha_is_penality_coeff] - 1 <= self.epsilon
        cond_alpha_other = np.abs(cond[alpha_other] - 1) < self.epsilon
        if np.sum(cond_alpha_is_zero) != np.sum(alpha_is_zero):
            return False
        if np.sum(cond_alpha_is_penality_coeff) != np.sum(alpha_is_penality_coeff):
            return False
        if np.sum(cond_alpha_other) != np.sum(alpha_other):
            return False
        return True

    def _select_alphas(self):
        cond = self.labels * self._cal_g(self.features)
        alpha_is_zero = np.abs(self.alpha) < self.epsilon
        alpha_is_penality_coeff = np.abs(self.alpha - self.penality_coeff) < self.epsilon
        alpha_other = (1 - (alpha_is_zero + alpha_is_penality_coeff)) > 0

        cond_alpha_is_zero = cond[alpha_is_zero] - 1 >= self.epsilon
        cond_alpha_is_penality_coeff = cond[alpha_is_penality_coeff] - 1 <= self.epsilon
        cond_alpha_other = np.abs(cond[alpha_other] - 1) < self.epsilon
        if np.sum(cond_alpha_other) != np.sum(alpha_other):
            _cond = copy.copy(cond)
            _cond[~alpha_other] = 1.0
            alpha_1 = np.argmax(np.abs(_cond - 1.0))
        elif np.sum(cond_alpha_is_zero) != np.sum(alpha_is_zero):
            _cond = copy.copy(cond)
            _cond[~alpha_is_zero] = 1.0
            alpha_1 = np.argmax(1 - _cond)
        elif np.sum(cond_alpha_is_penality_coeff) != np.sum(alpha_is_penality_coeff):
            _cond = copy.copy(cond)
            _cond[~alpha_is_penality_coeff] = 1.0
            alpha_1 = np.argmax(_cond - 1)
        alpha_2 = np.argmax(np.abs(self.E[alpha_1] - self.E))

        ##  alpha_1, alpha_2 is index of the selected alpha.
        return alpha_1, alpha_2

    def _clip_alphas(self, alpha_1, alpha_2, alpha_2_value):
        if self.labels[alpha_1] * self.labels[alpha_2] < 0:
            lower_bound = max([0, self.alpha[alpha_2] - self.alpha[alpha_1]])
            upper_bound = min([
                self.penality_coeff, self.penality_coeff + self.alpha[alpha_2] - self.alpha[alpha_1]
            ])
        elif self.labels[alpha_1] * self.labels[alpha_2] > 0:
            lower_bound = max([
                0, self.alpha[alpha_2] + self.alpha[alpha_1] - self.penality_coeff
            ])
            upper_bound = min([
                self.penality_coeff, self.alpha[alpha_2] + self.alpha[alpha_1]
            ])
        alpha_2_value = max([lower_bound, alpha_2_value])
        alpha_2_value = min([upper_bound, alpha_2_value])
        return alpha_2_value

    def train(self, features, labels):
        self._setup(features, labels)
        print(">>>> training start.")
        pbar = tqdm(total=self.max_iter)
        for iter_cnt in range(self.max_iter):
            ##  select alphas to be updated.
            alpha_1, alpha_2 = self._select_alphas()

            ##  calculate alpha_2's new value.
            eta = self.transform(self.features[alpha_1], self.features[alpha_1]) + \
                self.transform(self.features[alpha_2], self.features[alpha_2]) - \
                    2 * self.transform(self.features[alpha_1], self.features[alpha_2])
            alpha_2_value = self.alpha[alpha_2] + \
                (self.labels[alpha_2] * (self.E[alpha_1] - self.E[alpha_2])) / eta

            ##  clip alpha_2 and calculate alpha_1's new value
            alpha_2_value = self._clip_alphas(alpha_1, alpha_2, alpha_2_value)
            alpha_1_value = self.alpha[alpha_1] + \
                self.labels[alpha_1] * self.labels[alpha_2] * (self.alpha[alpha_2] - alpha_2_value)
            
            ##  calculate bias according alpha_1's new value and alpha_2's new value.
            bias_1 = - self.E[alpha_1] - self.labels[alpha_1] * \
                self.transform(self.features[alpha_1], self.features[alpha_1]) * (alpha_1_value - self.alpha[alpha_1]) - \
                    self.labels[alpha_2] * self.transform(self.features[alpha_2], self.features[alpha_1]) * \
                        (alpha_2_value - self.alpha[alpha_2]) + self.bias
            bias_2 = - self.E[alpha_2] - self.labels[alpha_1] * \
                self.transform(self.features[alpha_1], self.features[alpha_2]) * (alpha_1_value - self.alpha[alpha_1]) - \
                    self.labels[alpha_2] * self.transform(self.features[alpha_2], self.features[alpha_2]) * \
                        (alpha_2_value - self.alpha[alpha_2]) + self.bias

            ##  update alphas, bias and E
            if 0 < alpha_1_value < self.penality_coeff:
                self.bias = bias_1
            elif 0 < alpha_2_value < self.penality_coeff:
                self.bias = bias_2
            else:
                self.bias = (bias_1 + bias_2) / 2
            self.alpha[alpha_1] = alpha_1_value
            self.alpha[alpha_2] = alpha_2_value
            self.E[alpha_1] = self._cal_E(self.features[alpha_1], self.labels[alpha_1])
            self.E[alpha_2] = self._cal_E(self.features[alpha_2], self.labels[alpha_2])
            if self._satify_stop_condition():
                break
            pbar.update()
        pbar.close()
        self.weights = np.matmul(
            self.alpha * self.labels, self.features
        )
        print(">>>> training finished.")
        
    def predict(self, features):
        self.pred_features = features
        preds = np.matmul(
            self.alpha * self.labels, np.matmul(
                features, self.features.transpose(-1, -2)
            ).transpose(-1, -2)
        ) + self.bias
        preds[preds > 0] = 1
        preds[preds < 0] = -1
        self.preds = preds
        return preds

    def visualize(self, path, title="result", xlabel="x", ylabel="y"):
        # pos = np.concatenate([self.pred_features[self.preds > 0], self.features[self.labels > 0]])
        # neg = np.concatenate([self.pred_features[self.preds < 0], self.features[self.labels < 0]])
        pos = self.features[self.labels > 0]
        neg = self.features[self.labels < 0]
        # neg = self.pred_features[self.preds < 0]
        plt.scatter(pos[:, 0], pos[:, 1], s=30, c="blue", label="positive")
        plt.scatter(neg[:, 0], neg[:, 1], s=30, c="green", label="negative")
        pos_support_vector, neg_support_vector = self.get_support_vector()
        plt.scatter(
            pos_support_vector[:, 0], pos_support_vector[:, 1], \
                s=50, c="blue", label="positive support vector", edgecolors="black", \
                    facecolor="none", linewidths=1
        )
        plt.scatter(
            neg_support_vector[:, 0], neg_support_vector[:, 1], \
                s=50, c="green", label="negative support vector", edgecolors="black", \
                    facecolor="none", linewidths=1
        )
        x_min = np.min(self.pred_features[:, 0])
        x_max = np.max(self.pred_features[:, 0])
        x_min, x_max = x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)
        x = np.arange(x_min, x_max, (x_max - x_min) / 100)
        # x = np.array([i for i in range(x_min, x_max, (x_max - x_min) / 100)])
        y = - self.weights[0] / self.weights[1] * x - self.bias / self.weights[1]
        plt.plot(x, y)
        mean_support_vector = np.sum(pos_support_vector, axis=0) / pos_support_vector.shape[0]
        y = - self.weights[0] / self.weights[1] * x - self.bias / self.weights[1] - \
            (-self.weights[0] / self.weights[1] * mean_support_vector[0]-self.bias/self.weights[1] - mean_support_vector[1])
        plt.plot(x, y, linestyle="--")
        mean_support_vector = np.sum(neg_support_vector, axis=0) / neg_support_vector.shape[0]
        y = - self.weights[0] / self.weights[1] * x - self.bias / self.weights[1] - \
            (-self.weights[0] / self.weights[1] * mean_support_vector[0]-self.bias/self.weights[1] - mean_support_vector[1])
        plt.plot(x, y, linestyle="--")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(path)
        plt.close()
        
    def get_support_vector(self):
        index = (self.alpha > self.epsilon) & ((self.penality_coeff - self.alpha) > self.epsilon)
        # alpha_is_zero = np.abs(self.alpha) < self.epsilon
        # alpha_is_penality_coeff = np.abs(self.alpha - self.penality_coeff) < self.epsilon
        # alpha_other = (1 - (alpha_is_zero + alpha_is_penality_coeff)) > 0
        # index = self.labels > 0
        pos_support_vector = self.features[index & (self.labels > 0)]
        neg_support_vector = self.features[index & (self.labels < 0)]
        return pos_support_vector, neg_support_vector



def load_data(dataset="iris"):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

def svm(X_train, X_test, Y_train, Y_test, path):
    plt.figure(figsize=(10, 5))
    for i, C in enumerate([1, 100]):
        # "hinge" is the standard SVM loss
        clf = LinearSVC(C=C, loss="hinge", random_state=42).fit(X_train, Y_train)
        # obtain the support vectors through the decision function
        decision_function = clf.decision_function(X_train)
        # we can also calculate the decision function manually
        # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
        support_vector_indices = np.where((2 * Y_train - 1) * decision_function <= 1)[0]
        support_vectors = X_train[support_vector_indices]

        plt.subplot(1, 2, i + 1)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=30, cmap=plt.cm.Paired)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                             np.linspace(ylim[0], ylim[1], 50))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                    linestyles=['--', '-', '--'])
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                    linewidth=1, facecolors='none', edgecolors='k')
        plt.title("C=" + str(C))
    plt.tight_layout()
    plt.savefig(path)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    
    logger = Logger()
    from ml.configs.configs import cfg as cfg

    logger.log_info("loading data...")
    logger.record_item_time("load_data")
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=65)
    logger.log_info("done after {:.3f} seconds".format(logger.show_item_time("load_data")))

    # model = SVM(kernel="polynomial", penality_coeff=100)
    model = SVM(penality_coeff=100)
    
    logger.log_info("training...")
    logger.record_item_time("train")
    model.train(X_train, Y_train)
    logger.log_info("done after {:.3f} seconds".format(logger.show_item_time("train")))

    Y_preds = model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_preds)
    logger.log_info("accuracy: {:.5f}".format(accuracy))

    model.visualize(os.path.join(cfg.SVM.DIR, "custom_SVM.png"))

    svm(X_train, X_test, Y_train, Y_test, os.path.join(cfg.SVM.DIR, "sklearn_SVM"))

    logger.log_info("reach the end of program.")