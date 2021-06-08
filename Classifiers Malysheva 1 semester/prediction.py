import numpy as np
import pandas as pd
import joblib
from os.path import join
from sklearn.preprocessing import RobustScaler
import pickle

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

#еще сделать папку
def OB_GQ(data_dir, X, proba):

    try:
        X_train = np.load(join(data_dir, 'models/X_train_gq_dp.npy'))
        with open(join(data_dir, 'models/gb_gq_dp.pkl'), 'rb') as f:
            gb = pickle.load(f)
        
    except:
        print('Unable to open X_train_gq_dp.npy, gb_gq.pkl')
        exit(0)

    robust = RobustScaler()
    
    X_train = robust.fit_transform(X_train)
    X = robust.transform(X)
    if proba:
        pred = gb.predict_proba(X)[:, 0]
        np.save(join(data_dir, 'prediction_proba_GQ'), pred)
    else:
        pred = gb.predict(X)
        np.save(join(data_dir, 'prediction_GQ'), pred)
    return pred


def OB_ST(data_dir, X, proba):
    
    try:
        X_train = np.load(join(data_dir, 'models/X_train_ob_st.npy'))
        with open(join(data_dir, 'models/gb_ob_st.pkl'), 'rb') as f:
            gb = pickle.load(f)
    except:
        print('Unable to open X_train_st.npy, gb_ob_st.pkl')
        exit(0)

    
    robust = RobustScaler()

    X_train = robust.fit_transform(X_train)
    X = robust.transform(X)

    if proba:
        pred = gb.predict_proba(X)[:, 0]
    else:
        pred = gb.predict(X)
    np.save(join(data_dir, 'prediction_OB_ST'), pred)

    return pred
    

def ST(data_dir, X, proba):

    try:
        X_train = np.load(join(data_dir, 'models/X_train_st.npy'))
        with open(join(data_dir, 'models/gb_st.pkl'), 'rb') as f:
            gb = pickle.load(f)
    except:
        print('Unable to open X_train_st.npy, gb_st.pkl')
        exit(0)
    
    robust = RobustScaler()

    X_train = robust.fit_transform(X_train)
    X = robust.transform(X)

    if proba:
        pred = gb.predict_proba(X)[:, 0]
    else:
        pred = gb.predict(X)
    np.save(join(data_dir, 'prediction_ST'), pred)

    return pred

