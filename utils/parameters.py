"""This file contains the Param class which contains all the 
parameters needed for data loading, data processing, model training,
and model processing.

This class is imported in all codes required for training. Any
changes to parameters could be done collectively here.
"""

import torch

class Params:
    """
    Parameters for training models.
    """
    def __init__(self):
        # Model name -- '<type>_<raw/pretrained>_<input>_<version>'
        # CLS name format
        #self.MODEL_NAME = 'alexnetMap_cls_top5_v3.2.2'
        #self.MODEL_NAME = 'grConvMap_cls_top5_v1.0'
        # Grasp name format
        self.MODEL_NAME = 'multiAlexMap_top5_v1.5'
        #Seed instructions
        # 1=pretrained on grasp, then trained on cls
        # 2=pretrained on cls, then trained on grasp
        # 11=singletask model cls
        # 22 = single task model grasp
        
        # 31 = single task model cls v2 (trained post steven)
        # 32 = single task model grasp v2 (trained post steven)
        # 33 = single task model cls, pretrained on grasp v2
        # 34 = single task model grasp, pretrained on cls v2 
        # 43-47 + no_seed = base model
        # 203=depth input only
        # 301=rgb only 
        # i think 400 is trained with weight decay/L2 reg
        self.SEED = 32
        self.MODEL_NAME_SEED = self.MODEL_NAME + f"_{self.SEED}"
        self.CLS_MODEL_NAME = 'alexnetMap_cls_top5_v3.2.2'
        self.GRASP_MODEL_NAME = 'alexnetMap_grasp_top5_v3.2.2'

        # device: cpu / gpu
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() \
                                      else torch.device('cpu')
        self.LOSS_WEIGHT = 1.5
        # Training params
        self.NUM_CLASS = 5
        self.NUM_CHANNEL = 4
        self.OUTPUT_SIZE = 224  # 128 was used for training grCLS
        self.IMG_SIZE = (self.NUM_CHANNEL, self.OUTPUT_SIZE, self.OUTPUT_SIZE) 
        self.EPOCHS = 150
        self.LR = 5e-4
        self.BATCH_SIZE = 5
        self.TRAIN_VAL_SPLIT = 0.1
        self.DISTILL_ALPHA = 1.0
        self.TRAIN_TASK_REP_PATH = './data/task_rep/train/data_location.pickle'
        self.TEST_TASK_REP_PATH = './data/task_rep/test/data_location.pickle'
        
        # Shapley params
        self.TOP_K = 5
        self.DATA_TRUNCATION = 0.5
        self.LAYERS = ['rgb_features.0', 'features.0', 'features.4', 'features.7', 'features.10']

        # Paths
        self.DATA_PATH = 'data'
        self.TRAIN_PATH = 'data/top_5_compressed/train'
        self.TRAIN_PATH_ALT = 'data/top_5_compressed_old/train'
        self.TEST_PATH = 'data/top_5_compressed/test'
        self.TRAIN_PATH_SHUFFLE = 'data/data_shuffle/top_5_compressed/train'
        self.TEST_PATH_SHUFFLE = 'data/data_shuffle/top_5_compressed/test'
        self.TEST_PATH_ALT = 'data/top_5_compressed_old/test'
        self.LABEL_FILE = 'cls_top_5.txt'

        self.MODEL_PATH = 'trained-models'
        self.MODEL_WEIGHT_PATH = 'trained-models/%s/%s_final.pth' % (self.MODEL_NAME, self.MODEL_NAME)
        self.MODEL_WEIGHT_PATH_SEED = 'trained-models/%s/%s_%s_final.pth' % (self.MODEL_NAME_SEED, self.MODEL_NAME, self.SEED)
        self.CLS_MODEL_PATH = 'trained-models/%s/%s_epoch%s.pth' % (self.CLS_MODEL_NAME, self.CLS_MODEL_NAME, self.EPOCHS)
        self.GRASP_MODEL_PATH = 'trained-models/%s/%s_epoch%s.pth' % (self.GRASP_MODEL_NAME, self.GRASP_MODEL_NAME, self.EPOCHS)
        self.CLS_WEIGHT_PATH = 'trained-models/alexnetMap_cls.pth'
        self.GRASP_WEIGHT_PATH = 'trained-models/alexnetMap_grasp.pth'
        
        self.MODEL_LOG_PATH = 'trained-models/%s_%s' % (self.MODEL_NAME, self.SEED)
        self.LOG_PATH = 'logs'
