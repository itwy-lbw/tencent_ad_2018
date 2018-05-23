from baseclassifier import BaseClassifier
import lightgbm as lgb
#from parms import RIDGE_PARMS as lp
import pandas as pd
#from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from parms import *

class PAC(BaseClassifier):
    def __init__(self,TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='pac',USE_TINY=False,RANDOMSTATE=2018):
        super(PAC, self).__init__(
                    TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name,USE_TINY,RANDOMSTATE)
        '''In Ridge, only 'sag' solver can currently fit the intercept when X is sparse.'''
        self.clf=PassiveAggressiveClassifier(n_iter=50, tol=1e-3)

    def trainWithEva(self,trainval_x):
        '''fit the data with evalidation'''
        train_x, valid_x, train_y, valid_y = train_test_split(\
                            trainval_x,self.trainval['label'],\
                            test_size=0.1, random_state=self.randomstate)
        self.clf.fit(train_x,train_y)
        pred = self.clf.decision_function(valid_x)
        #print(valid_y,pred)
        score=metrics.roc_auc_score(valid_y, pred)
        print("%s on valid set accuracy:   %0.5f" % (self.name,score))
        return score

    def predict(self,test_x=None,model_path=None):
        if model_path is not None:
            self.load_model(model_path)
        if test_x is None:
            _,test_x=self.feature_engineering()
        #self.clf.decision_function(test_x)
        #print(pd.read_csv(self.ds.TEST),self.ds.TEST)
        pre=pd.read_csv(self.ds.TEST)
        #print(test_x.shape,pre.shape)
        pre['score'] = self.clf.decision_function(test_x)
        pre['score'] = pre['score'].apply(lambda x: float('%.6f' % x))
        return pre

if __name__ == '__main__':
    '''Tiny'''
    bc=PAC(TRAINVALTEST_DENSE_X_TINY,TRAINVALTEST_DENSE_X_NAMES_TINY,\
                    TRAINVAL_SPARSE_X_TINY,TRAINVAL_SPARSE_X_NAMES_TINY,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE_TINY,\
                    TEST_MERGE,TEST,name='pac',USE_TINY=True)
    
    #print type(train_X),train_X.shape,val_X.shape     
    '''normal'''
    #bc=PAC(TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
    #                TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
    #                TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
    #                UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
    #                TEST_MERGE,TEST,name='pac',USE_TINY=False)
    bc.validate()
        
        
