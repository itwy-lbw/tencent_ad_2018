from baseclassifier import BaseClassifier
import lightgbm as lgb
#from parms import RIDGE_PARMS as lp
import pandas as pd
from sklearn.linear_model import Perceptron


from parms import *

class PT(BaseClassifier):
    def __init__(self,TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='perceptron',USE_TINY=False,RANDOMSTATE=2018):
        super(PT, self).__init__(
                    TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name,USE_TINY,RANDOMSTATE)
        '''In Ridge, only 'sag' solver can currently fit the intercept when X is sparse.'''
        self.clf=Perceptron(n_iter=50, tol=1e-3)


if __name__ == '__main__':
    '''Tiny'''
    #bc=Perceptron(TRAINVALTEST_DENSE_X_TINY,TRAINVALTEST_DENSE_X_NAMES_TINY,\
    #                TRAINVAL_SPARSE_X_TINY,TRAINVAL_SPARSE_X_NAMES_TINY,\
    #                TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
    #                UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE_TINY,\
    #                TEST_MERGE,TEST,name='perceptron',USE_TINY=True)
    
    #print type(train_X),train_X.shape,val_X.shape     
    '''normal'''
    bc=PT(TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='perceptron',USE_TINY=False)
    #bc.predict_submission()
    bc.validate()
    
        
        
