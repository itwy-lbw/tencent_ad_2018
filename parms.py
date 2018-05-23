#USE_TINY=False
DATA_ROOT='data'

'''original files'''
UF_VW=DATA_ROOT+'/userFeature.data'
ADF=DATA_ROOT+'/adFeature.csv'
TRAINVAL=DATA_ROOT+'/train.csv'
TEST=DATA_ROOT+'/test2.csv'

'''most used format'''
UF_CSV=DATA_ROOT+'/userFeature.csv'

'''randomseed used at fully process and merger files used before FE'''
RANDOMSTATE=2018
TRAINVAL_MERGE_TINY=DATA_ROOT+'/trainval_merge_tiny_'+str(RANDOMSTATE)+'.csv'
TRAINVAL_MERGE=DATA_ROOT+'/trainval_merge.csv'
TEST_MERGE=DATA_ROOT+'/test_merge.csv'

FEA_ROOT='features'
'''saved features'''
TRAINVALTEST_DENSE_X=FEA_ROOT+'/trainvaltest_dense_normal'
TRAINVALTEST_DENSE_X_NAMES=FEA_ROOT+'/names_trainvaltest_dense_normal'
TRAINVALTEST_DENSE_X_TINY=FEA_ROOT+'/trainvaltest_dense_tiny'+str(RANDOMSTATE)
TRAINVALTEST_DENSE_X_NAMES_TINY=FEA_ROOT+'/names_trainvaltest_dense_tiny'+str(RANDOMSTATE)
TRAINVAL_SPARSE_X=FEA_ROOT+'/trainval_sparse_normal'
TRAINVAL_SPARSE_X_NAMES=FEA_ROOT+'/names_trainval_sparse_normal'
TRAINVAL_SPARSE_X_TINY=FEA_ROOT+'/trainval_sparse_tiny'+str(RANDOMSTATE)
TRAINVAL_SPARSE_X_NAMES_TINY=FEA_ROOT+'/names_trainval_sparse_tiny'+str(RANDOMSTATE)
TEST_SPARSE_X=FEA_ROOT+'/test_sparse'
TEST_SPARSE_X_NAMES=FEA_ROOT+'/names_test_sparse'




TRAINFX=DATA_ROOT+'/train_X_'+str(RANDOMSTATE)+'.npz'
TRAINFY=DATA_ROOT+'/train_y_'+str(RANDOMSTATE)+'.npy'
VALFX=DATA_ROOT+'/val_X_'+str(RANDOMSTATE)+'.npz'
VALFY=DATA_ROOT+'/val_y_'+str(RANDOMSTATE)+'.npy'
TESTFX=DATA_ROOT+'/test_X.npz'
#TESTFY=DATA_ROOT+'/test_y.npz'
TRAINVALFX=DATA_ROOT+'/trainval_X.npz'
TRAINVALFY=DATA_ROOT+'/trainval_y.npy'
TRAINFX_TINY=DATA_ROOT+'/train_X_'+str(RANDOMSTATE)+'_tiny.npz'
TRAINFY_TINY=DATA_ROOT+'/train_y_'+str(RANDOMSTATE)+'_tiny.npy'
VALFX_TINY=DATA_ROOT+'/val_X_'+str(RANDOMSTATE)+'_tiny.npz'
VALFY_TINY=DATA_ROOT+'/val_y_'+str(RANDOMSTATE)+'_tiny.npy'
TESTFX_TINY=DATA_ROOT+'/test_X._tinynpz'
#TESTFY=DATA_ROOT+'/test_y.npz'
TRAINVALFX_TINY=DATA_ROOT+'/trainval_X_tiny.npz'
TRAINVALFY_TINY=DATA_ROOT+'/trainval_y_tiny.npy'
GENERATEF_TINY=DATA_ROOT+'/generateFeatures_tiny.csv'
GENERATEF=DATA_ROOT+'/generateFeatures.csv'


'''parms for classifier'''
LGB_PARMS={
    'boosting_type':'gbdt',
    'num_leaves':31, 
    'reg_alpha':0.0, 
    'reg_lambda':1,
    'max_depth':-1, 
    'n_estimators':10000, 
    'objective':'binary',
    'subsample':0.7, 
    'colsample_bytree':0.7, 
    'subsample_freq':1,
    'learning_rate':0.05, 
    'min_child_weight':100, 
    'random_state':RANDOMSTATE, 
    'n_jobs':-1,
    'eval_metric':'auc',
    'early_stopping_rounds':'500'
}
