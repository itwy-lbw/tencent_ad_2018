from baseclassifier import BaseClassifier
#import lightgbm as lgb
import xgboost as xgb
from parms import LGB_PARMS as lp
import pandas as pd
from sklearn.model_selection import train_test_split
from parms import *
from sklearn import metrics
import copy

class XGB(BaseClassifier):
    def __init__(self,TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='xgb',USE_TINY=False,RANDOMSTATE=2018):
        super(XGB, self).__init__(TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name,USE_TINY,RANDOMSTATE)
        '''
        self.clf=XGB.XGBMClassifier(
            boosting_type=lp['boosting_type'], num_leaves=lp['num_leaves'], \
            reg_alpha=lp['reg_alpha'], reg_lambda=lp['reg_lambda'],\
            max_depth=lp['max_depth'], n_estimators=lp['n_estimators'], objective=lp['objective'],\
            subsample=lp['subsample'], colsample_bytree=lp['colsample_bytree'], subsample_freq=lp['subsample_freq'],\
            learning_rate=lp['learning_rate'], min_child_weight=lp['min_child_weight'], random_state=lp['random_state'],\
            n_jobs=lp['n_jobs']
            )
        '''
        self.clf = xgb.XGBClassifier(n_estimators=10000,subsample=0.8,colsample_bytree=0.8)
        '''if want to used different feature files , set self.f_name'''
        self.f_name='base'
    '''
    def trainWithoutEva(self):
        self.feature_engineering()
        self.clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], \
                    eval_metric=lp['eval_metric'], verbose=100, \
                    early_stopping_rounds=lp['early_stopping_rounds'])
    '''
    def trainWithoutEva(self,trainval_x):
        '''fit the data without evalidation'''
        self.clf.fit(trainval_x,self.ds.trainval_y,\
                    eval_set=[(trainval_x, self.ds.trainval_y)],\
                    eval_metric='auc',early_stopping_rounds=100)
        #self.resolve_clf_and_save_dict()
    def trainWithEva(self,trainval_x):
        '''fit the data with evalidation'''
        train_x, valid_x, train_y, valid_y = train_test_split(\
                            trainval_x,self.ds.trainval_y,\
                            test_size=0.1, random_state=self.randomstate)
        self.clf.fit(train_x,train_y,\
                    eval_set=[(train_x, train_y),(valid_x,valid_y)],\
                    eval_metric='auc',early_stopping_rounds=100)
        pred = self.clf.predict_proba(valid_x)[:,1]
        #print(valid_y,pred)
        score=metrics.roc_auc_score(valid_y, pred)
        print("%s on valid set accuracy:   %0.5f" % (self.name,score))
        return score

if __name__ == '__main__':
    '''Tiny'''
    #bc=XGB(TRAINVALTEST_DENSE_X_TINY,TRAINVALTEST_DENSE_X_NAMES_TINY,\
    #                TRAINVAL_SPARSE_X_TINY,TRAINVAL_SPARSE_X_NAMES_TINY,\
    #                TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
    #                UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE_TINY,\
    #                TEST_MERGE,TEST,name='sgb',USE_TINY=True)
    
    #print type(train_X),train_X.shape,val_X.shape     
    '''normal'''
    bc=XGB(TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='XGB',USE_TINY=False)
    bc.validate()
    #bc.predict_submission()
        

