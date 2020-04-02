
from easydict import EasyDict as edict
import os


cfg = {
    "DIR": "/media/yiqunchen/Data/Documents/MyUniversity/Academic/MachineLearning/Homework/ml", 
    "PERCEPTION": {
        # "DATA_PATH": see code below.
    }, 
    "BAYES":{
        # "DATA_PATH": see code below
    }, 
    "DECISION_TREE": {
        # 
    }, 
    "SVM": {
        #   "DIR"
    }
}

cfg = edict(cfg)

cfg.PERCEPTION.DATA_PATH = os.path.join(
    cfg.DIR, 
    "data/train_binary.csv"
)
assert os.path.exists(cfg.PERCEPTION.DATA_PATH), \
    "path {} not found, please check it.".format(cfg.PERCEPTION.DATA_PATH)

cfg.BAYES.DATA_PATH = os.path.join(
    cfg.DIR, 
    "data/train.csv"
)
assert os.path.exists(cfg.BAYES.DATA_PATH), \
    "path {} not found, please check it.".format(cfg.BAYES.DATA_PATH)

cfg.DECISION_TREE.DATA_PATH = os.path.join(
    cfg.DIR, 
    "data/train.csv"
)
assert os.path.exists(cfg.DECISION_TREE.DATA_PATH), \
    "path {} not found, please check it.".format(cfg.DECISION_TREE.DATA_PATH)

cfg.SVM.DIR = os.path.join(
    cfg.DIR, "ml", "SVM"
)