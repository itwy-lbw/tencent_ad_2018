from baseclassifier import BaseClassifier
import lightgbm as lgb
#from parms import RIDGE_PARMS as lp
import pandas as pd
#from sklearn.linear_model import RidgeClassifier
import xlearn as xl
from sklearn.model_selection import train_test_split
from sklearn import metrics


from parms import *

class FFM(BaseClassifier):
    def __init__(self,TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='ffm',USE_TINY=False,RANDOMSTATE=2018):
        super(FFM, self).__init__(
                    TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name,USE_TINY,RANDOMSTATE=2018)
        '''In Ridge, only 'sag' solver can currently fit the intercept when X is sparse.'''
        self.clf=xl.FFMModel(task='binary', 
                        lr=0.2, 
                        epoch=10, 
                        reg_lambda=0.002,
                        metric='auc')
def trainWithoutEva(self,trainval_x):
    '''fit the data without evalidation'''
    self.clf.fit(trainval_x,self.trainval['label'],\
                 eval_set=[trainval_x, self.trainval['label']],\
                 is_lock_free=False)
    #self.resolve_clf_and_save_dict()
def trainWithEva(self,trainval_x):
    '''fit the data with evalidation'''
    train_x, valid_x, train_y, valid_y = train_test_split(\
                        trainval_x,self.trainval['label'],\
                        test_size=0.1, random_state=self.randomstate)
    self.clf.fit(train_x, train_y,
                 eval_set=[valid_x, valid_y],
                 is_lock_free=False)
    pred = self.clf.predict(valid_x)[:,1]
    #print(valid_y,pred)
    score=metrics.roc_auc_score(valid_y, pred)
    print("%s on valid set accuracy:   %0.5f" % (self.name,score))
    return score    


if __name__ == '__main__':
    '''Tiny'''
    bc=FFM(TRAINVALTEST_DENSE_X_TINY,TRAINVALTEST_DENSE_X_NAMES_TINY,\
                    TRAINVAL_SPARSE_X_TINY,TRAINVAL_SPARSE_X_NAMES_TINY,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE_TINY,\
                    TEST_MERGE,TEST,name='ffm',USE_TINY=True)
    
    #print type(train_X),train_X.shape,val_X.shape     
    '''normal'''
    #bc=FFM(TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
    #                TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
    #                TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
    #                UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
    #                TEST_MERGE,TEST,name='ffm',USE_TINY=False)
    bc.validate()
        
        
