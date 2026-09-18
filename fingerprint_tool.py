# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:21:57 2026

@author: user
"""
import sys
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
#import chembl_structure_pipeline
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
import joblib
import pickle
#from IPython.display import HTML
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from molvs import standardize_smiles
#from KRFingerprints import *
import warnings
from rdkit.Chem import AllChem, MACCSkeys
from argparse import ArgumentParser
from argparse import BooleanOptionalAction
import math
import pickle

warnings.filterwarnings('ignore')


#initialdir=os.getcwd()

parser = ArgumentParser()
parser.add_argument('-i','--input', type=str, default=None, help='.csv file with SMILES notations', required=True)
parser.add_argument('-m', '--model',type=str, default=None, help='machine learning tool (KNN/RF/SVM/GB/AB/MLP/XGB/ET/CB/DT)',required=True)
parser.add_argument('-p', '--parameter', type=str, default=None, help='.csv file with ML parameters',required=True)
parser.add_argument('-fp', '--fingerprint',type=str, default='morgan', help='morgan/maccs/rdkit/kr (default=morgan)',required=False)
parser.add_argument('-r', '--radius',type=int, default=4, help='radius for Morgan (default=4)',required=False)
parser.add_argument('-nb', '--nBits', type=int, default=1024, help='nBits for Morgan (default=1024)',required=False)
parser.add_argument('-uf',"--useFeatures", action='store_true', help="Use features for Morgan, default=False", required=False)
parser.add_argument('-uc', "--useChirality", action='store_true', help='Use Chirality for Morgan, default=False',required=False)
parser.add_argument('-ts', '--testSize', type=float, default=0.2, help='test set size, default=0.2',required=False)
parser.add_argument('-rs', '--randomState',type=float, default=42, help='random_state, default=42',required=False) 
parser.add_argument('-cv', '--crossV',type=float, default=5, help='cross_validation fold, default=5',required=False) 
parser.add_argument('-sm', '--saveModel',action='store_true', help="Model saving, default=True", required=False) 
parser.add_argument('-sl', '--smilesLabel',type=str, default='SMILES', help='label for SMILES column, default=SMILES',required=False)
parser.add_argument('-sf', '--saveFiles',action='store_true', help="Training/Test data saving, default=True", required=False)

                  

args = parser.parse_args()
colhead=args.smilesLabel

def write_versions(filer):
    import sklearn
    import xgboost
    import lightgbm
    filer.write('Scikit-learn version: '+ str(sklearn.__version__)+'\n')
    filer.write('XGBoost version: '+ str(xgboost.__version__)+'\n')
    filer.write('LightGBM version: '+ str(lightgbm.__version__)+'\n')
    filer.write('Numpy version: '+str(np.__version__)+'\n')
    filer.write('Pandas version: '+str(pd.__version__)+'\n')
    filer.write('Python version:' + str(sys.version)+'\n')
    filer.write('Rdkit version'+str(rdkit.__version__)+'\n')
    filer.write('\n')
    
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def rdkit_numpy_convert(fp_vs):
    output = []
    for f in fp_vs:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def convert_morgan(df,radius,nBits,useFeatures,useChirality):
    ls=[]
    for smile in df[colhead]:
        smiles=standardize_smiles(smile)
        ls.append(Chem.MolFromSmiles(smile))
    fp = [AllChem.GetMorganFingerprintAsBitVect(m, radius=radius,nBits=nBits,useFeatures=useFeatures,useChirality = useChirality) for m in ls]
    x = rdkit_numpy_convert(fp)
    pd.DataFrame(x).to_csv('morgan_{}_{}_uc{}_{}.csv'.format(radius,nBits,useFeatures,df.shape[0]))
    return x

def convert_other(df, option):
    ls=[]
    for smile in df[colhead]:
        smiles=standardize_smiles(smile)
        ls.append(Chem.MolFromSmiles(smile))
    if option=='rdkit':
       fp=[Chem.RDKFingerprint(m) for m in ls]
       x = rdkit_numpy_convert(fp)
       pd.DataFrame(x).to_csv('rdkit_{}.csv'.format(df.shape[0]))
    elif option=='maccs':
       fp =  [MACCSkeys.GenMACCSKeys(x) for x in ls]
       x = rdkit_numpy_convert(fp)
       pd.DataFrame(x).to_csv('maccs_{}.csv'.format(df.shape[0]))
    return x

def convert_kr(df):
    fp = KRFingerprints.GenerateKRFingerprintsToDataFrame(df,colhead)
    fp2 = fp.iloc[:,3:]
    x = np.array(fp2)
    fp2.to_csv('kr_{}.csv'.format(df.shape[0]))
    return x
    

def predict(y,ypr):
    Kappa = metrics.cohen_kappa_score(y, ypr, weights='linear')
    confusion_matrix_CV_GBM=metrics.confusion_matrix(y,ypr)
    TN, FP, FN, TP = confusion_matrix_CV_GBM.ravel()
    SE = TP/(TP+FN)
    SP = TN/(TN+FP)
    BA = (SE + SP)/2
    MCC=(TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return BA,MCC

def parse_value(v):
    v = str(v).strip()
    print(v)

    # Boolean
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    # None
    if v.lower() == "none":
        return None

    # Integer
    try:
        if "." in v:
            if v.split('.')[1]=='0':
                if int(v.split('.')[0])>0:
                   return int(v.split('.')[0])
            elif float(v)>0:
                   return float(v)
        else: 
            return int(v)
                    
    except:
        pass

    # Float

    # String
    return v


def csv_to_param_grid(csv_file):
    df = pd.read_csv(csv_file)

    param_grid = {}

    for col in df.columns:
        values = df[col].dropna().unique()
        param_grid[col] = [parse_value(v) for v in values] 

    return param_grid
    

def selected():
    param_grid=csv_to_param_grid(args.parameter)
    print(param_grid)
    rd=int(args.randomState)
    file3=pd.read_csv(args.parameter)
    if args.model=='RF':
        estimator1=RandomForestClassifier(random_state=rd)
        rn='RF'
    elif args.model=='KNN':
        estimator1=KNeighborsClassifier()
        rn='KNN'       
    elif args.model=='AB':
         estimator1 = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=rd), random_state=rd)
         rn='AB'

    elif args.model=='SVM':
        estimator1=SVC()
        rn='SVM'        
    elif args.model=='GB':
         estimator1=GradientBoostingClassifier(verbose=0, random_state=rd)
         rn='GB'   
    elif args.model=='MLP':
         estimator1=MLPClassifier(max_iter=7000, random_state=rd)
         rn='MLP'
         #param_grid ['hidden_layer_sizes']=[(50,50)]
         #param_grid= {"hidden_layer_sizes": [(50, 50)], "activation": ["identity", "logistic", "tanh", "relu"],
                      #'alpha': [0.0001, 0.001, 0.01, 0.1],'learning_rate': ['constant','adaptive', 'invscaling']}

         if 'hidden_layer_sizes' in file3.columns:
               lst = file3['hidden_layer_sizes'].values.tolist()
               result = [tuple(int(x) for x in s.split(',') if x) for s in lst]
               param_grid['hidden_layer_sizes']=result
         else:
               pass
         print(param_grid)
    elif args.model=='XGB':
         estimator1=XGBClassifier(objective="reg:squarederror",random_state=rd)
         rn='XGB'
    elif args.model=='ET':
         estimator1=ExtraTreesClassifier(random_state=rd)
         rn='ET'
    #elif args.model=='LGB':
         #estimator1=LGBMClassifier(objective='regression',random_state=rd)
         #rn='LGB' 
    elif args.model=='DT':
         estimator1=DecisionTreeClassifier(random_state=rd)
         rn='DT'
    elif args.model=='CB':
         estimator1=CatBoostClassifier(random_state=rd)
         rn='CB'  
    else:
        pass
    return estimator1,param_grid,rn

def gbm(x_tr,x_ts,y_tr,y_ts,ml,pm):
    seed = 42
    cv = StratifiedKFold(n_splits=int(args.crossV), shuffle=True, random_state=seed)
    #param_grid ={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    #param_grid=param_grid
    model = GridSearchCV(ml,pm, n_jobs=2, cv=cv, verbose=1)
    model.fit(x_tr, y_tr)
    best_clf_GBM = model.best_estimator_
    print(best_clf_GBM)
    y_pred_CV_GBM = cross_val_predict(best_clf_GBM, x_tr, y_tr, cv=cv)
    y_ts_pred=best_clf_GBM.predict(x_ts)
    BA_tr,MCC_tr=predict(y_tr, y_pred_CV_GBM)
    BA_ts, MCC_ts=predict(y_ts, y_ts_pred)
    return BA_tr, BA_ts, MCC_tr, MCC_ts,best_clf_GBM

def main_morgan(tr,ts,radius,nBits,useFeatures,useChirality,ml,pm):
    #tr,ts=train_test_split(df,random_state=i, test_size=0.2)
    x_tr=convert_morgan(tr,radius,nBits,useFeatures,useChirality)
    x_ts=convert_morgan(ts,radius,nBits,useFeatures,useChirality)
    y_tr=tr.iloc[:,-1:]
    y_ts=ts.iloc[:,-1:]
    BA_tr,BA_ts,MCC_tr,MCC_ts,bcg=gbm(x_tr,x_ts,y_tr,y_ts,ml,pm)
    return BA_tr, BA_ts, MCC_tr, MCC_ts,bcg

def main_other(tr,ts,option,ml,pm):
    #tr,ts=train_test_split(df,random_state=i, test_size=0.2)
    x_tr=convert_other(tr,option)
    x_ts=convert_other(ts,option)
    y_tr=tr.iloc[:,-1:]
    y_ts=ts.iloc[:,-1:]
    BA_tr,BA_ts,MCC_tr,MCC_ts,bcg=gbm(x_tr,x_ts,y_tr,y_ts,ml,pm)
    return BA_tr,BA_ts,MCC_tr,MCC_ts,bcg

def main_kr(tr,ts,ml,pm):
    #tr,ts=train_test_split(df,random_state=i, test_size=0.2)
    x_tr=convert_kr(tr)
    x_ts=convert_kr(ts)
    y_tr=tr.iloc[:,2:3]
    y_ts=ts.iloc[:,2:3]
    BA_tr,BA_ts,MCC_tr,MCC_ts,bcg=gbm(x_tr,x_ts,y_tr,y_ts,ml,pm)
    return BA_tr,BA_ts,MCC_tr,MCC_ts,bcg

def morgan_type(radius,useFeatures):
    if (radius==2) & (useFeatures==False):
        mt='ECFP4'
    elif (radius==3) & (useFeatures==False):
        mt='ECFP6'
    elif (radius==4) & (useFeatures==False):
        mt='ECFP8'
    elif (radius==2) & (useFeatures==True):
        mt='FCFP4'
    elif (radius==3) & (useFeatures==True):
        mt='FCFP6'
    elif (radius==4) & (useFeatures==True):
        mt='FCFP8'
    return mt
    


def process():
    filer = open("versions.txt","w")
    write_versions(filer)
    df=pd.read_csv(args.input)   #upload input file here
    tr,ts=train_test_split(df,test_size=args.testSize, random_state=int(args.randomState))
    if args.saveFiles==True:
       tr.to_csv('Training_data.csv', index=False)
       ts.to_csv('Test_data.csv', index=False)
    option=args.fingerprint
    radius=args.radius
    nBits=args.nBits 
    useFeatures=args.useFeatures 
    useChirality=args.useChirality
    ml,pm,rn=selected()
    if option=='morgan':
       a_,b_,c_,d_,e_=main_morgan(tr,ts,radius,nBits,useFeatures,useChirality,ml,pm) 
       mt= morgan_type(radius,useFeatures)
       nb=nBits
       uc= useChirality         
    elif option=='rdkit':
       a_,b_,c_,d_,e_=main_other(tr,ts,'rdkit',ml,pm)
       mt='rdkit'
       nb=None
       uc= None 
    elif option=='maccs':
       a_,b_,c_,d_,e_=main_other(tr,ts,'maccs',ml,pm)
       mt='maccs'
       nb=None
       uc= None
    elif option=='kr':
       a_,b_,c_,d_,e_=main_kr(tr,ts,ml,pm)
       mt='kr'
       nb=None
       uc= None
    dic={}
    dic['Fingerprint']=[mt]
    dic['nBits']=[nb]
    dic['UseChirality']=[uc]
    dic['model']=[rn]
    dic['BA_tr']=[round(a_,3)]
    dic['BA_ts']=[round(b_,3)]
    dic['MCC_tr']=[round(c_,3)]
    dic['MCC_ts']=[round(d_,3)]
    print(dic)
    result=pd.DataFrame(dic)
    mrs=int(args.randomState)
    result.to_csv('result_{}_{}_RS{}.csv'.format(mt, rn,mrs), index=False)
    if args.saveModel==True:
        pname='best_model_{}_{}_RS{}.pkl'.format(mt,rn,mrs)
        pickle.dump(e_, open(pname, 'wb'))
    else:
        pass
    
    
if __name__ == '__main__':
   process()
