#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 二值化
def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img

def Train(trainset,train_labels):
    prior_probability = np.zeros(class_num)                         # 先验概率
    conditional_probability = np.zeros((class_num,feature_len,2))   # 条件概率

    # 计算先验概率及条件概率
    for i in range(len(train_labels)):
        img = binaryzation(trainset[i])     # 图片二值化
        label = train_labels[i]

        prior_probability[label] += 1

        for j in range(feature_len):
            conditional_probability[label][j][img[j]] += 1

    # 将概率归到[1.10001]
    for i in range(class_num):
        for j in range(feature_len):

            # 经过二值化后图像只有0，1两种取值
            pix_0 = conditional_probability[i][j][0]
            pix_1 = conditional_probability[i][j][1]

            # 计算0，1像素点对应的条件概率
            probalility_0 = (float(pix_0)/float(pix_0+pix_1))*1000000 + 1
            probalility_1 = (float(pix_1)/float(pix_0+pix_1))*1000000 + 1

            conditional_probability[i][j][0] = probalility_0
            conditional_probability[i][j][1] = probalility_1

    return prior_probability,conditional_probability

# 计算概率
def calculate_probability(img,label):
    probability = int(prior_probability[label])

    for i in range(len(img)):
        probability *= int(conditional_probability[label][i][img[i]])

    return probability

def Predict(testset,prior_probability,conditional_probability):
    predict = []

    for img in testset:

        # 图像二值化
        img = binaryzation(img)

        max_label = 0
        max_probability = calculate_probability(img,0)

        for j in range(1,10):
            probability = calculate_probability(img,j)

            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)

    return np.array(predict)


class_num = 10
feature_len = 784

if __name__ == '__main__':

    from ml.configs.configs import cfg
    from ml.utils.metrics import Logger

    logger = Logger()

    logger.log_info("reading data...")
    logger.record_item_time("reading_data")

    raw_data = pd.read_csv(cfg.BAYES.DATA_PATH,header=0)
    data = raw_data.values

    imgs = data[0::,1::]
    labels = data[::,0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
    # print train_features.shape
    # print train_features.shape

    logger.log_info("done after {:.3f} seconds".format(logger.show_item_time("reading_data")))

    logger.log_info("training...")
    logger.record_item_time("training")
    prior_probability,conditional_probability = Train(train_features,train_labels)
    logger.log_info("done after {:.3f} seconds".format(logger.show_item_time("training")))

    logger.log_info("predicting...")
    logger.record_item_time("evaluation")
    test_predict = Predict(test_features,prior_probability,conditional_probability)
    logger.log_info("done after {:.3f} seconds".format(logger.show_item_time("evaluation")))

    score = accuracy_score(test_labels,test_predict)
    logger.log_info("accuracy: {}".format(score))
    logger.log_info("reach the end of program.")