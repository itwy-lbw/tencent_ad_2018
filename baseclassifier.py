#coding=utf-8
import os 
import pandas as pd
import numpy as np
from scipy import sparse
from dataset import DataSet
#from sklearn.preprocessing import LabelBinarizer
from parms import *
import scipy.special as special
import copy
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
import shutil
import datetime

    
class BaseClassifier(object):
    def __init__(self,TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                    TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
                    TEST_MERGE,TEST,name='base',USE_TINY=False,RANDOMSTATE=2018):
        self.name=name
        self.f_name='base'

        self.use_tiny=USE_TINY
        self.randomstate=RANDOMSTATE
        '''data used by classifier'''
        self.ds=DataSet(UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,TEST_MERGE,TEST,\
                TRAINVALTEST_DENSE_X,TRAINVAL_SPARSE_X,TRAINVALTEST_DENSE_X_NAMES,\
                TRAINVAL_SPARSE_X_NAMES,TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                USE_TINY,RANDOMSTATE)
        '''
        if self.use_tiny:
            self.trainval,self.test=self.ds.load_data()
            self.data=self.trainval
        else:
            self.trainval,self.test=self.ds.load_data()
            self.test['label']=-1
            self.data=pd.concat([self.trainval,self.test],ignore_index=True,sort=False)
        print('using normal dataset,size:',self.data.shape,self.trainval.shape,self.test.shape)
        self.n_trainval=len(self.trainval)
        '''

        self.clf='abstract'
        self.arc_name=None
        self.currentModel_path=None
        
    def trainWithoutEva(self,trainval_x):
        '''fit the data without evalidation'''
        self.clf.fit(trainval_x,self.ds.trainval_y)
        #self.resolve_clf_and_save_dict()
    def trainWithEva(self,trainval_x):
        '''fit the data with evalidation'''
        train_x, valid_x, train_y, valid_y = train_test_split(\
                            trainval_x,self.ds.trainval_y,\
                            test_size=0.1, random_state=self.randomstate)
        self.clf.fit(train_x,train_y)
        pred = self.clf.predict_proba(valid_x)[:,1]
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
        #print(self.clf.predict_proba(test_x)[:,1].shape)
        pre['score'] = self.clf.predict_proba(test_x)[:,1]
        pre['score'] = pre['score'].apply(lambda x: float('%.6f' % x))
        return pre

    def feature_engineering(self):
        return self.ds.feature_engineering(self.f_name)

    def predict_submission(self):
        '''api for full dataset training'''
        if not self.use_tiny:
            trainval_x,test_x=self.feature_engineering()
            self.trainWithoutEva(trainval_x)
            self.save_model()
            pre=self.predict(test_x)
            self.create_submission(pre)
            self.resolve_clf_and_save_dict()
        else:
            trainval_x=self.feature_engineering()
            score=self.trainWithEva(trainval_x)
            self.resolve_clf_and_save_dict(score)
    def validate(self):
        '''api for validation dataset training'''
        if not self.use_tiny:
            trainval_x,test_x=self.feature_engineering()
            score=self.trainWithEva(trainval_x)
            self.save_model()
            pre=self.predict(test_x)
            self.create_submission(pre)
            #self.save_model()
            #pre=self.predict(test_x)
            #self.create_submission(pre)
            self.resolve_clf_and_save_dict(score)
        else:
            trainval_x=self.feature_engineering()
            score=self.trainWithEva(trainval_x)
            self.resolve_clf_and_save_dict(score)
    def create_submission(self,df):
        df.to_csv('tmp/submission.csv',index=False)
        t=datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
        self.arc_name='submissions/'+self.name+'_'+self.f_name+t
        shutil.make_archive(self.arc_name, 'zip', 'tmp')
    def save_model(self):
        t=datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
        self.currentModel_path='models/'+self.name+'_'+self.f_name+'_'+str(self.use_tiny)+t+'.model'
        joblib.dump(self.clf, self.currentModel_path)
    def load_model(self,model_path):
        self.currentModel_path=model_path
        self.clf=joblib.load(model_path)
    def resolve_clf(self):
        line=str(self.clf)
        print type(line)
        dic={}
        dic['classifier_name']=line.split('(')[0]
        contents=line.split('(')[1].split(')')[0].replace('\n','').replace(' ','').split(',')
        for c in contents:
            dic[c.split('=')[0]]=c.split('=')[1]
        print(str(dic))
        return dic
    def resolve_clf_and_save_dict(self,score=None):
        dic=self.resolve_clf()
        dic['use_tiny']=self.use_tiny
        dic['feature_group']=self.f_name

        dic['randseed']=self.randomstate
        #dic['tranval_nums']=self.n_trainval
        if self.arc_name is not None:
            dic['submission_arch_name']=self.arc_name
        if self.currentModel_path is not None:
            dic['model_path']=self.currentModel_path
        if score is not None:
            dic['auc_value']=score
        try:
            dic['stop_rounds']=self.clf.best_iteration_
        except:
            pass
        with open('results.dic', 'a') as f:
            f.writelines(str(dic)+'\n')
if __name__ == '__main__':
    '''Tiny'''
    bc=BaseClassifier(TRAINVALTEST_DENSE_X_TINY,TRAINVALTEST_DENSE_X_NAMES_TINY,\
                    TRAINVAL_SPARSE_X_TINY,TRAINVAL_SPARSE_X_NAMES_TINY,\
                    TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
                    UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE_TINY,\
                    TEST_MERGE,TEST,name='base',USE_TINY=True)
    
    #print type(train_X),train_X.shape,val_X.shape     
    '''normal'''
    #bc=BaseClassifier(TRAINVALTEST_DENSE_X,TRAINVALTEST_DENSE_X_NAMES,\
    #                TRAINVAL_SPARSE_X,TRAINVAL_SPARSE_X_NAMES,\
    #                TEST_SPARSE_X,TEST_SPARSE_X_NAMES,\
    #                UF_VW,ADF,TRAINVAL,UF_CSV,TRAINVAL_MERGE,\
    #                TEST_MERGE,TEST,name='base',USE_TINY=False)
    bc.feature_engineering()
        
