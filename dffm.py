from baseclassifier import BaseClassifier
import lightgbm as lgb
#from parms import RIDGE_PARMS as lp
import pandas as pd
#from sklearn.linear_model import RidgeClassifier
import xlearn as xl
from sklearn.model_selection import train_test_split
from sklearn import metrics
from DeepFM import DeepFM
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from DataReader import FeatureDictionary, DataParser
from parms import *
from sklearn.externals import joblib

class DFFM(BaseClassifier):
    def __init__(self,TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='deepffm',USE_TINY=False,RANDOMSTATE=2018):
        super(DFFM, self).__init__(
                    TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name,USE_TINY,RANDOMSTATE=2018)
        '''In Ridge, only 'sag' solver can currently fit the intercept when X is sparse.'''
        dfm_params = {
                    "use_fm": True,
                    "use_deep": True,
                    "embedding_size": 8,
                    "dropout_fm": [1.0, 1.0],
                    "deep_layers": [32, 32],
                    "dropout_deep": [0.5, 0.5, 0.5],
                    "deep_layers_activation": tf.nn.relu,
                    "epoch": 72,
                    "batch_size": 1024,
                    "learning_rate": 0.001,
                    "optimizer_type": "adam",
                    "batch_norm": 1,
                    "batch_norm_decay": 0.995,
                    "l2_reg": 0.01,
                    "verbose": True,
                    "eval_metric": roc_auc_score,
                    "random_seed":2018
                }
        dfTrainVal,dfTest=self.ds.load_TrainVal_Test()
        fd = FeatureDictionary(dfTrain=dfTrainVal, dfTest=dfTest,
                               numeric_cols=[],
                               ignore_cols=[])
        data_parser = DataParser(feat_dict=fd)
        #dfTrain_x, dfVal_x, dfTrain_y, dfVal_y =train_test_split(dfTrainVal.drop(['label'],axis=1)\
        #                                        ,dfTrainVal['label'],test_size=0.1, random_state=self.randomstate)
        #dfTrain=pd.DataFrame([dfTrain_x,dfTrain_y])
        #dfVal=pd.DataFrame([dfVal_x,dfVal_y])
        #print dfTrain.shape
        devideline=int(0.9*len(dfTrainVal))
        dfTrain=dfTrainVal[:devideline]
        dfVal=dfTrainVal[devideline:]
        Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
        Xi_valid, Xv_valid, y_valid = data_parser.parse(df=dfVal, has_label=True)
        Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

        dfm_params["feature_size"] = fd.feat_dim
        dfm_params["field_size"] = len(Xi_train[0])
        self.clf=DeepFM(**dfm_params)

        # fit a DeepFM model
        self.clf.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid, early_stopping=True, refit=True)

        y_pred=self.clf.predict(Xi_test, Xv_test)    
        ids_test["label"]= y_pred
        ids_test.to_csv('submission_dffm.csv', index=False, float_format="%.5f")
        joblib.dump(self.clf, 'saved_model.model')
if __name__ == '__main__':
    '''Tiny'''
    #bc=DFFM(TRAINVALTEST_DENSE_X_TINY,TRAINVALTEST_DENSE_X_NAMES_TINY,\
    #                TRAINVAL_SPARSE_X_TINY,TRAINVAL_SPARSE_X_NAMES_TINY,\
    #                TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
    #                UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE_TINY,\
    #                TEST_MERGE,TEST,name='dffm',USE_TINY=True)
    
    #print type(train_X),train_X.shape,val_X.shape     
    '''normal'''
    bc=DFFM(TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='dffm',USE_TINY=False)
    #bc.validate()
        
        
