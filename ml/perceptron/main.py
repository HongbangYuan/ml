
"""
Author  
Time    
Notes   You can increase the self.max_iteration to increase accuracy.
"""

import pandas as pd
import numpy as np
import cv2
import random
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 50

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)

        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])

    def dual_train(self, features, labels):
        """
        Info    train the perceptron.
        Args    features(list): the training input.
                labels(list): training labels.
        Returns None.
        Note    I don't calculate the Gram Matrix as memory is finite.
        """
        ##  initialize the alpha and beta
        self.a = np.zeros((len(features)))
        self.b = 0
        ##  we use numpy to perform matrix product.
        np_features = np.array(features)        
        np_labels = 2 * np.array(labels) - 1
        ##  correct_counter can be ignored.
        correct_counter = 0
        iteration = 0
        ##  tqdm can draw a processing bar in terminal.
        pbar = tqdm(total=self.max_iteration)
        while iteration < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            ##  get one sample according the random number index.
            y = np_labels[index]
            x = features[index]
            ##  calculate the products between x and all inputs.
            x_dots = np.array([np.matmul(x_j, x) for x_j in features])
            ##  predict the result
            pred = np.sum(self.a * np_labels * x_dots) + self.b
            if pred * y > 0:
                ##  do nothing if classify is correct
                correct_counter += 1
            else:
                ##  update the correspond alpha_i and bias b
                self.a[index] += self.learning_step
                self.b = self.b + self.learning_step * y
            ##  don't forget update the iteration monitor.
            iteration += 1
            pbar.set_description("iteration {}".format(iteration))
            pbar.update()
        pbar.close()
        ##  store the weight help the predict process.
        self.weight = np.matmul((self.a*np_labels), np_features)
        # print(">>>> weight shape: {}".format(self.weight.shape))
        self.bias = np.sum(self.a*np_labels)
        # print(">>>> bias shape: {}".format(self.bias.shape))
            
    def dual_predict(self, features):
        """
        Info    predict the inputs.
        Args    features(list): the inputs.
        Returns preds(list): classes predicted by perceptron.
        """
        preds = []
        np_features = np.array(features)
        pbar = tqdm(total=len(features))
        for x in features:
            ##  predict the class.
            preds.append(
                int((np.matmul(self.weight, x)+self.bias)>0)
            )
            pbar.set_description("predicting")
            pbar.update()
        pbar.close()
        ##  just for debugging.
        assert len(preds) == len(features), "predict failed, please check dimension"
        return preds

    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':

    from ml.utils.metrics import Logger
    from ml.configs.configs import cfg

    logger = Logger()
    logger.log_info("reading data...")
    logger.record_item_time("read_data")

    raw_data = pd.read_csv(cfg.PERCEPTION.DATA_PATH, header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape
    logger.log_info("finished after {:.3f} seconds.".format(logger.show_item_time("read_data")))

    logger.log_info("training model...")
    logger.record_item_time("training")
    p = Perceptron()
    # p.train(train_features, train_labels)
    p.dual_train(train_features, train_labels)

    logger.log_info("done after {:.3f} seconds".format(logger.show_item_time("training")))

    logger.log_info("predicting...")
    logger.record_item_time("evaluation")
    # test_predict = p.predict(test_features)
    test_predict = p.dual_predict(test_features)
    logger.log_info("done after {:.3f} seconds".format(logger.show_item_time("evaluation")))

    score = accuracy_score(test_labels, test_predict)
    logger.log_info("accuracy: {}".format(score))
