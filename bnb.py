from baseclassifier import BaseClassifier
import lightgbm as lgb
#from parms import RIDGE_PARMS as lp
import pandas as pd
from sklearn.naive_bayes import BernoulliNB


from parms import *

class BNB(BaseClassifier):
    def __init__(self,TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='bnb',USE_TINY=False,RANDOMSTATE=2018):
        super(BNB, self).__init__(
                    TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name,USE_TINY,RANDOMSTATE=2018)
        '''In Ridge, only 'sag' solver can currently fit the intercept when X is sparse.'''
        self.clf=BernoulliNB(alpha=0.1,fit_prior=False)
        self.f_name='base_norm'


if __name__ == '__main__':
    '''Tiny'''
    bc=BNB(TRAINVALTEST_DENSE_X_TINY,TRAINVALTEST_DENSE_X_NAMES_TINY,\
                    TRAINVAL_SPARSE_X_TINY,TRAINVAL_SPARSE_X_NAMES_TINY,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE_TINY,\
                    TEST_MERGE,TEST,name='bnb',USE_TINY=True)
    
    #print type(train_X),train_X.shape,val_X.shape     
    '''normal'''
    #bc=BNB(TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
    #                TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
    #                TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
    #                UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
    #                TEST_MERGE,TEST,name='bnb',USE_TINY=False)
    #bc.predict_submission()
    bc.validate()
        
        
