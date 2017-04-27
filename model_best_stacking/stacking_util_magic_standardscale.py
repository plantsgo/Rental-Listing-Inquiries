#encoding=utf8
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost
import lightgbm

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB



##############################################################分类####################################################
def stacking(clf,train_x,train_y,test_x,clf_name,class_num=3):
    train=np.zeros((train_x.shape[0],class_num))
    test=np.zeros((test_x.shape[0],class_num))
    test_pre=np.empty((folds,test_x.shape[0],class_num))
    cv_scores=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf","ada","gb","et","lr","knn","mnb","ovr","gnb"]:
            clf.fit(tr_x,tr_y)
            pre=clf.predict_proba(te_x)
            train[test_index]=pre
            test_pre[i,:]=clf.predict_proba(test_x)
            cv_scores.append(log_loss(te_y, pre))
        elif clf_name in ["lsvc"]:
            clf.fit(tr_x,tr_y)
            pre=clf.decision_function(te_x)
            train[test_index]=pre
            test_pre[i,:]=clf.decision_function(test_x)
            cv_scores.append(log_loss(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'objective': 'multi:softprob',
                      'eval_metric': 'mlogloss',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12,
                      "num_class": class_num
                      }

            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(test_matrix,ntree_limit=model.best_ntree_limit)
                train[test_index]=pre
                test_pre[i, :]= model.predict(z, ntree_limit=model.best_ntree_limit)
                cv_scores.append(log_loss(te_y, pre))
        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            #z = clf.Dataset(test_x, label=te_y)
            #z=test_x
            params = {
                      'boosting_type': 'gbdt',
                      #'boosting_type': 'dart',
                      'objective': 'multiclass',
                      'metric': 'multi_logloss',
                      'min_child_weight': 1.5,
                      'num_leaves': 2**5,
                      'lambda_l2': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'learning_rate': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12,
                      "num_class": class_num,
                      'silent': True,
                      }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(te_x,num_iteration=model.best_iteration)
                train[test_index]=pre
                test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration)
                cv_scores.append(log_loss(te_y, pre))
        elif clf_name in ["nn"]:
            from keras.layers import Dense, Dropout, BatchNormalization,SReLU
            from keras.optimizers import SGD,RMSprop
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.utils import np_utils
            from keras.regularizers import l2
            from keras.models import Sequential
            clf = Sequential()
            clf.add(Dense(64, input_dim=tr_x.shape[1],activation="relu", W_regularizer=l2()))
            #clf.add(SReLU())
            #clf.add(Dropout(0.2))
            clf.add(Dense(64,activation="relu",W_regularizer=l2()))
            #clf.add(SReLU())
            #clf.add(Dense(64, activation="relu", W_regularizer=l2()))
            # model.add(Dropout(0.2))
            clf.add(Dense(class_num, activation="softmax"))
            clf.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            reduce = ReduceLROnPlateau(min_lr=0.0002,factor=0.05)
            clf.compile(optimizer="rmsprop", loss="categorical_crossentropy")
            clf.fit(tr_x, tr_y,
                      batch_size=640,
                      nb_epoch=1000,
                      validation_data=[te_x, te_y],
                      callbacks=[early_stopping,reduce])
            pre=clf.predict_proba(te_x)
            train[test_index]=pre
            test_pre[i,:]=clf.predict_proba(test_x)
            cv_scores.append(log_loss(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print "%s now score is:"%clf_name,cv_scores
        with open("score.txt","a") as f:
            f.write("%s now score is:"%clf_name+str(cv_scores)+"\n")
    test[:]=test_pre.mean(axis=0)
    print "%s_score_list:"%clf_name,cv_scores
    print "%s_score_mean:"%clf_name,np.mean(cv_scores)
    with open("score.txt", "a") as f:
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,class_num),test.reshape(-1,class_num)

def rf(x_train, y_train, x_valid):
    randomforest = RandomForestClassifier(n_estimators=1200, max_depth=20, n_jobs=-1, random_state=2017, max_features="auto",verbose=1)
    rf_train, rf_test = stacking(randomforest, x_train, y_train, x_valid,"rf")
    return rf_train, rf_test,"rf"

def ada(x_train, y_train, x_valid):
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking(adaboost, x_train, y_train, x_valid,"ada")
    return ada_train, ada_test,"ada"

def gb(x_train, y_train, x_valid):
    gbdt = GradientBoostingClassifier(learning_rate=0.04, n_estimators=100, subsample=0.8, random_state=2017,max_depth=5,verbose=1)
    gbdt_train, gbdt_test = stacking(gbdt, x_train, y_train, x_valid,"gb")
    return gbdt_train, gbdt_test,"gb"

def et(x_train, y_train, x_valid):
    extratree = ExtraTreesClassifier(n_estimators=1200, max_depth=35, max_features="auto", n_jobs=-1, random_state=2017,verbose=1)
    et_train, et_test = stacking(extratree, x_train, y_train, x_valid,"et")
    return et_train, et_test,"et"

def ovr(x_train, y_train, x_valid):
    est=RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=2017, max_features="auto",
                               verbose=1)
    ovr = OneVsRestClassifier(est,n_jobs=-1)
    ovr_train, ovr_test = stacking(ovr, x_train, y_train, x_valid,"ovr")
    return ovr_train, ovr_test,"ovr"

def xgb(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking(xgboost, x_train, y_train, x_valid,"xgb")
    return xgb_train, xgb_test,"xgb"

def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking(lightgbm, x_train, y_train, x_valid,"lgb")
    return xgb_train, xgb_test,"lgb"

def gnb(x_train, y_train, x_valid):
    gnb=GaussianNB()
    gnb_train, gnb_test = stacking(gnb, x_train, y_train, x_valid,"gnb")
    return gnb_train, gnb_test,"gnb"

def lr(x_train, y_train, x_valid):
    logisticregression=LogisticRegression(n_jobs=-1,random_state=2017,C=0.1,max_iter=200)
    lr_train, lr_test = stacking(logisticregression, x_train, y_train, x_valid, "lr")
    return lr_train, lr_test, "lr"

def fm(x_train, y_train, x_valid):
    pass


def lsvc(x_train, y_train, x_valid):

    #linearsvc=SVC(probability=True,kernel="linear",random_state=2017,verbose=1)
    #linearsvc=SVC(probability=True,kernel="linear",random_state=2017,verbose=1)
    linearsvc=LinearSVC(random_state=2017)
    lsvc_train, lsvc_test = stacking(linearsvc, x_train, y_train, x_valid, "lsvc")
    return lsvc_train, lsvc_test, "lsvc"

def knn(x_train, y_train, x_valid):
    #pca = PCA(n_components=10)
    #pca.fit(x_train)
    #x_train = pca.transform(x_train)
    #x_valid = pca.transform(x_valid)

    kneighbors=KNeighborsClassifier(n_neighbors=200,n_jobs=-1)
    knn_train, knn_test = stacking(kneighbors, x_train, y_train, x_valid, "knn")
    return knn_train, knn_test, "knn"

def nn(x_train, y_train, x_valid):
    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train)
    nn_train, nn_test = stacking("", x_train, y_train, x_valid, "nn")
    return nn_train, nn_test, "nn"
###########################################################################################################

#####################################################回归##################################################

###########################################################################################################

def stacking_reg(clf,train_x,train_y,test_x,clf_name):
    train=np.zeros((train_x.shape[0],1))
    test=np.zeros((test_x.shape[0],1))
    test_pre=np.empty((folds,test_x.shape[0],1))
    cv_scores=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf","ada","gb","et","lr","lsvc","knn"]:
            clf.fit(tr_x,tr_y)
            pre=clf.predict(te_x).reshape(-1,1)
            train[test_index]=pre
            test_pre[i,:]=clf.predict(test_x).reshape(-1,1)
            cv_scores.append(mean_squared_error(te_y, pre))
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'eval_metric': 'rmse',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12
                      }
            num_round = 10000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(test_matrix,ntree_limit=model.best_ntree_limit).reshape(-1,1)
                train[test_index]=pre
                test_pre[i, :]= model.predict(z, ntree_limit=model.best_ntree_limit).reshape(-1,1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            #z = clf.Dataset(test_x, label=te_y)
            #z=test_x
            params = {
                      'boosting_type': 'gbdt',
                      'objective': 'regression_l2',
                      'metric': 'mse',
                      'min_child_weight': 1.5,
                      'num_leaves': 2**5,
                      'lambda_l2': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'learning_rate': 0.03,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 12,
                      'silent': True,
                      }
            num_round = 10000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(te_x,num_iteration=model.best_iteration).reshape(-1,1)
                train[test_index]=pre
                test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape(-1,1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["nn"]:
            from keras.layers import Dense, Dropout, BatchNormalization
            from keras.optimizers import SGD,RMSprop
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.utils import np_utils
            from keras.regularizers import l2
            from keras.models import Sequential
            clf = Sequential()
            clf.add(Dense(64, input_dim=tr_x.shape[1], activation="relu", W_regularizer=l2()))
            # model.add(Dropout(0.2))
            clf.add(Dense(64, activation="relu", W_regularizer=l2()))
            # model.add(Dropout(0.2))
            clf.add(Dense(1))
            clf.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            reduce = ReduceLROnPlateau(min_lr=0.0002,factor=0.05)
            clf.compile(optimizer="rmsprop", loss="mse")
            clf.fit(tr_x, tr_y,
                      batch_size=640,
                      nb_epoch=5000,
                      validation_data=[te_x, te_y],
                      callbacks=[early_stopping, reduce])
            pre=clf.predict(te_x).reshape(-1,1)
            train[test_index]=pre
            test_pre[i,:]=clf.predict(test_x).reshape(-1,1)
            cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError("Please add new clf.")
        print "%s now score is:"%clf_name,cv_scores
        with open("score.txt","a") as f:
            f.write("%s now score is:"%clf_name+str(cv_scores)+"\n")
    test[:]=test_pre.mean(axis=0)
    print "%s_score_list:"%clf_name,cv_scores
    print "%s_score_mean:"%clf_name,np.mean(cv_scores)
    with open("score.txt", "a") as f:
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,1),test.reshape(-1,1)

def rf_reg(x_train, y_train, x_valid):
    randomforest = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs=-1, random_state=2017, max_features="auto",verbose=1)
    rf_train, rf_test = stacking_reg(randomforest, x_train, y_train, x_valid,"rf")
    return rf_train, rf_test,"rf_reg"

def ada_reg(x_train, y_train, x_valid):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid,"ada")
    return ada_train, ada_test,"ada_reg"

def gb_reg(x_train, y_train, x_valid):
    gbdt = GradientBoostingRegressor(learning_rate=0.04, n_estimators=100, subsample=0.8, random_state=2017,max_depth=5,verbose=1)
    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid,"gb")
    return gbdt_train, gbdt_test,"gb_reg"

def et_reg(x_train, y_train, x_valid):
    extratree = ExtraTreesRegressor(n_estimators=600, max_depth=35, max_features="auto", n_jobs=-1, random_state=2017,verbose=1)
    et_train, et_test = stacking_reg(extratree, x_train, y_train, x_valid,"et")
    return et_train, et_test,"et_reg"

def lr_reg(x_train, y_train, x_valid):
    lr_reg=LinearRegression(n_jobs=-1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, "lr")
    return lr_train, lr_test, "lr_reg"

def xgb_reg(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking_reg(xgboost, x_train, y_train, x_valid,"xgb")
    return xgb_train, xgb_test,"xgb_reg"

def lgb_reg(x_train, y_train, x_valid):
    lgb_train, lgb_test = stacking_reg(lightgbm, x_train, y_train, x_valid,"lgb")
    return lgb_train, lgb_test,"lgb_reg"

def nn_reg(x_train, y_train, x_valid):
    nn_train, nn_test = stacking_reg("", x_train, y_train, x_valid, "nn")
    return nn_train, nn_test, "nn_reg"
##########################################################################################################

#####################################################获取数据##############################################

###########################################################################################################
def get_data():
    '''
    train=pd.read_csv("train_python.csv")
    valid=pd.read_csv("test_python.csv")
    train = train.fillna(-1).replace(np.inf, -1)
    valid = valid.fillna(-1).replace(np.inf, -1)

    x_train=train.drop(["interest_level"],axis=1).values
    y_train=train["interest_level"].values
    x_valid = valid.values
    '''
    all = pd.read_csv("all20.csv")
    all = all.fillna(-1).replace(np.inf, -1)
    categorical = ["display_address", "manager_id", "building_id", "street_address","man_bui_id"]
    for f in categorical:
        if all[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(all[f].values))
            all[f] = lbl.transform(list(all[f].values))
    all=all.replace({"interest_level":{"high":0,"medium":1,"low":2,"nnnn":3},
                     "description":{0:"o"}
                     })
    train = all[all.interest_level != 3].copy()
    valid = all[all.interest_level == 3].copy()

    y_train=train["interest_level"]

    train_num=train.shape[0]

    tfidf = CountVectorizer(stop_words='english', max_features=100)
    all_sparse=tfidf.fit_transform(all["features"].values.astype('U')).toarray()
    tr_sparse = all_sparse[:train_num]
    te_sparse = all_sparse[train_num:]
    #print tfidf.get_feature_names()

    x_train = train.drop(["interest_level",'listing_id',"features","description","manager_features","same"],axis=1).values
    x_valid = valid.drop(["interest_level",'listing_id',"features","description","manager_features","same"],axis=1).values

    x_train = np.concatenate([x_train,tr_sparse],axis=1)
    x_valid = np.concatenate([x_valid,te_sparse],axis=1)


    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    print x_train.shape,x_valid.shape


    train_df=pd.DataFrame(x_train)
    test_df=pd.DataFrame(x_valid)
    train_df["listing_id"]=train["listing_id"].values
    test_df["listing_id"]=valid["listing_id"].values
    train_df.to_csv("best_model_train.csv",index=None)
    test_df.to_csv("best_model_test.csv",index=None)


    return x_train,y_train,x_valid,train,valid

from sklearn.feature_selection import SelectFromModel
def select_feature(clf,x_train,x_valid):
    clf.fit(x_train, y_train)
    model = SelectFromModel(clf, prefit=True, threshold="mean")

    print x_train.shape
    x_train = model.transform(x_train)
    x_valid = model.transform(x_valid)
    print x_train.shape

    return x_train,x_valid
if __name__=="__main__":
    np.random.seed(1)
    x_train, y_train, x_valid, train, valid = get_data()

    #选择重要特征
    '''
    clf=GradientBoostingClassifier()
    x_train,x_valid=select_feature(clf,x_train,x_valid)
    train_df=pd.DataFrame(x_train)
    test_df=pd.DataFrame(x_valid)
    train_df["listing_id"]=train["listing_id"].values
    test_df["listing_id"]=valid["listing_id"].values
    train_df.to_csv("best_model_train_top_feature.csv",index=None)
    test_df.to_csv("best_model_test_top_feature.csv",index=None)
    '''

    train_listing = train["listing_id"].values
    test_listing = valid["listing_id"].values

    folds = 5
    seed = 1
    kf = KFold(x_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)

    #############################################选择模型###############################################
    #
    #
    #
    #clf_list = [xgb,nn,knn,gb,rf,et,lr,ada_reg,rf_reg,gb_reg,et_reg,xgb_reg,nn_reg]
    clf_list = [xgb_reg,lgb_reg,nn_reg,lgb,xgb,lr,rf,et,gb,nn]
    #
    #
    #
    column_list = []
    train_data_list=[]
    test_data_list=[]
    for clf in clf_list:
        train_data,test_data,clf_name=clf(x_train,y_train,x_valid)
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        if "reg" in clf_name:
            ind_num=1
        else:
            ind_num=3
        for ind in range(ind_num):
            column_list.append("standardscaler_%s_%s" % (clf_name, ind))

    train = np.concatenate(train_data_list, axis=1)
    test = np.concatenate(test_data_list, axis=1)

    train = pd.DataFrame(train)
    train.columns = column_list
    train["level"] = pd.Series(y_train)
    train["listing_id"] = train_listing

    test = pd.DataFrame(test)
    test.columns = column_list
    test["listing_id"] = test_listing

    train.to_csv("magic_standardscaler_train_stacking.csv", index=None)
    test.to_csv("magic_standardscaler_test_stacking.csv", index=None)

