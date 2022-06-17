
# %%
#!pip install scikit-learn==0.18.2
#!pip install tensorflow
#%%
from turtle import color
import warnings
warnings.filterwarnings("ignore")
#%%
#!pip install xgboost
# conda install py-xgboost
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
from keras.models import Sequential
from keras.layers import Dense
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
from datetime import datetime
from datetime import date
import warnings
import eland as ed
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from decouple import config
#!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
from pandas_profiling import ProfileReport
import scipy.stats as stats
from numpy import percentile
pd.set_option('display.max_columns', None)

#%%
#profile = ProfileReport(dataset, title='Pandas Profiling Report', explorative=True)
def find_best_params(X_train, X_test, y_train, y_test,model,param_grid):
    grid_cv = GridSearchCV(model, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")
    grid_cv.fit(X_train, y_train)
    print('Best score:', grid_cv.best_score_)
    print('Best params:', grid_cv.best_params_)
    return grid_cv

def perf_measure(y_actual, y_pred, X_test): #canviar nom
    TP = []
    FP = []
    TN = []
    FN = []

    for i in range(len(y_pred)):
        a=0
        b=0
        c=0
        d=0
        if y_actual[i]==1 and y_pred[i]==1:
            TP.append(i)
        if y_pred[i]==1 and y_actual[i]==0:
            FP.append(i)
        if y_actual[i]==0 and y_pred[i]==0:
            TN.append(i)
        if y_pred[i]==0 and y_actual[i]==1:
            FN.append(i)

    return X_test.iloc[TP], X_test.iloc[FP], X_test.iloc[TN], X_test.iloc[FN]

def plot_ROCAUC(model_upn, y_test,X_test):
    baseline_roc_auc = roc_auc_score(y_test, model_upn.predict(X_test))
    fprB, tprB, thresholdsB = roc_curve(y_test, model_upn.predict_proba((X_test))[:,1])

    plt.figure()
    plt.plot(fprB, tprB, label='GBM Baseline (area = %0.2f)' % baseline_roc_auc)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    return ''

def train_model_and_performance_eval(model,train_x,test_x,train_y,test_y, cols, cutoff = 0.5, cf = 'coefficients'):
    %matplotlib inline
    model.fit(train_x,train_y)

    predictions = model.predict(test_x)
    probabilities = model.predict_proba(test_x)
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    cv_score = cross_val_score(model, test_x, test_y, cv=cv, scoring='roc_auc')
    print('Mean cv score:',np.mean(cv_score),'Std cv score:',np.std(cv_score),'Min cv score:',np.min(cv_score),'Max cv score:',np.max(cv_score))


    predictions = np.where(probabilities[:,1]>= cutoff,1,0)

    print("Accuracy score(training): {0:.3f}".format(model.score(train_x, train_y)))
    print("Accuracy score(validation): {0:.3f}".format(model.score(test_x, test_y)))
    print("\nModel Report")

    print("AUC Score (Train?): %f" % roc_auc_score(test_y, probabilities[:,1]))

    try:
        if   cf == "coefficients" :
            coefficients  = pd.DataFrame(model.coef_.ravel())
        elif cf == "features" :
            coefficients  = pd.DataFrame(model.feature_importances_)

        column_df     = pd.DataFrame(cols)
        coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                                right_index= True, how = "left"))
        coef_sumry.columns = ["coefficients","features"]
        coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)


        coef_sumry.reset_index(inplace=True)
        coef_sumry['colors'] = ['red' if x < 0 else 'green' for x in coef_sumry['coefficients']]
        coef_sumry


        plt.figure(figsize=(14,10), dpi= 80)
        plt.hlines(y=coef_sumry.index, xmin=0, xmax=coef_sumry.coefficients, color=coef_sumry.colors, alpha=0.4, linewidth=5)


        plt.gca().set(ylabel='$Features$', xlabel='$Coefficients$')
        plt.yticks(coef_sumry.index, coef_sumry.features, fontsize=12)
        plt.title('Feature Importance', fontdict={'size':20})
        plt.grid(linestyle='--', alpha=0.5)
    except:
        pass

    print(f'{model}\n\nClassification report:\n{classification_report(test_y,predictions)}')
    print(f'\nAccuracy Score: {accuracy_score(test_y,predictions)}')


    plot_confusion_matrix(model, test_x, test_y, cmap=plt.cm.Reds)
    plt.grid(False)

    model_and_real_values=pd.DataFrame({"real_value":predictions, "predicted_value":probabilities[:,1]})
    plot_dist(churn=model_and_real_values[model_and_real_values.real_value==1], no_churn=model_and_real_values[model_and_real_values.real_value==0],attr="predicted_value")



    plt.show()

    plot_ROCAUC(model, test_y,test_x)

    return perf_measure(test_y, predictions, test_x), probabilities,predictions



def separate_predict(upselling_model, test_companies, oversample=False):
    # Getting columns to be used when training
    cols = [i for i in upselling_model.columns if i not in ("target")]

    n_ones = Counter(upselling_model['target'])[1]
    n_test_ones = n_ones*0.25
    n_train_ones = n_ones*0.75

    n_train_zeros = (n_train_ones*75)/25
    n_test_zeros = (n_test_ones*75)/25

    zeros_test = test_companies[test_companies['target'] == 0]

    zeros_up = upselling_model[upselling_model['target'] == 0]
    ones_up = upselling_model[upselling_model['target'] == 1]


    train_ones, test_ones = train_test_split(ones_up, train_size = int(n_train_ones), shuffle=True) #0.3




    train_zeros, _ = train_test_split(zeros_up, train_size = int(n_train_zeros) , shuffle=True) #0.3
    test_companies_ = test_companies[(~test_companies.index.isin(train_zeros.index))]

    zeros_test = test_companies_[test_companies_['target'] == 0]
    _, test_zeros = train_test_split(zeros_test, test_size = int(n_test_zeros) , shuffle=True) #0.3


    train = train_zeros.append(train_ones)
    test = test_zeros.append(test_ones)


    if oversample == True:
        train_x, train_y = oversampling(train[cols],train[["target"]])
    else:
        train_x = train[cols].reset_index(drop=True)
        train_y = train["target"]
    companies_test = test.index
    val_x = test[cols].reset_index(drop=True)
    val_y = test["target"].values.ravel()

    return train_x, val_x, train_y, val_y, cols, companies_test



#Initializing the MLPClassifier
def MLP_model(X_train, y_train):
    classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    classifier.fit(X_train, y_train)
    return classifier

def GBT_classifier(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(n_estimators=100,random_state=0)
    grid_cv = find_best_params(X_train, X_test, y_train, y_test,clf,param_grid)
    #Once we find the best parameters
    final_cl = XGBClassifier(
    **grid_cv.best_params_,
    )
    final_cl.fit(X_train, y_train)
    return clf

def oversampling(upselling_X,upselling_Y):
    y = np.array(upselling_Y['target'])
    counter = Counter(y)
    print(counter)

    # transform the dataset, definition and SMOTE hyperparameter
    oversampling = SMOTE() #sampling_strategy=0.1, k_neighbors=10
    # application to create the new dataset with fraud samples augmented
    newX, newY = oversampling.fit_resample(upselling_X, upselling_Y)# summarize the new class distribution
    y_over = np.array(newY['target'])
    counter = Counter(y_over)
    print(counter)

    return newX,newY

def corr_matrix(dataset):
    %matplotlib inline
    #dataset = upselling[upselling['target']==1]

    corrMatrix = dataset.corr()
    plt.figure(figsize=(40, 40))
    sns.heatmap(corrMatrix, annot=True)
    plt.show()

def joint_plots(vars,data,high):
    dataset = upsample(data)
    dataset = dataset[(dataset[vars[0]]<high) & (dataset[vars[1]]<high)]
    dataset = dataset[vars]
    #sns.jointplot(x=var_x, y=var_y, data=dataset, color='seagreen', hue='target')
    sns.pairplot(dataset, hue="target", height=5.5, diag_kind="kde", palette=["lightcoral", "grey"])

    plt.show()

def allvars_description(dataset):
    %matplotlib inline
    data = dataset
    for i in data.columns:
        if i =='target' or i =='country' or i =='actual_plan':
            continue
        descriptions(data, i)

def descriptions(dataset, column,low=None, high=None):
    %matplotlib inline
    dataset = upsample(dataset)
    print(dataset[dataset['target']==0][column].describe())
    print(dataset[dataset['target']==1][column].describe())

    # % matplotlib inline
    if low == None and high == None:
        r = dataset[[column,'target']]
    else:
        if low != None and high != None:
            r = dataset[(dataset[column]>low)& (dataset[column]<high)][[column,'target']]

    sns.histplot(data=r,x=column,hue='target', palette=["plum", "skyblue"],binwidth= 1,multiple="layer") #import normal

    #sns.histplot(data=r,x=column,hue='target', palette=["plum", "skyblue"],binwidth= 2000,multiple="layer") #import normal
    #sns.histplot(data=r,x=column,hue='target', palette=["plum", "skyblue"],binwidth= 0.1,multiple="layer", log_scale=True) #import log

    #sns.histplot(data=r,x=column,hue='target', palette=["plum", "skyblue"],binwidth= 3,multiple="layer") #acum_0 normal
    #sns.histplot(data=r,x=column,hue='target', palette=["plum", "skyblue"],binwidth= .1,multiple="layer", log_scale=True) #acum log

    #sns.histplot(data=r,x=column,hue='target', palette=["plum", "skyblue"],binwidth= 10,multiple="layer") #days normal

    #sns.histplot(data=r,x=column,hue='target', palette=["plum", "skyblue"],binwidth= 18,multiple="layer")  #days last trans normal
    #sns.histplot(data=r,x=column,hue='target', palette=["plum", "skyblue"],binwidth= .05,multiple="layer", log_scale=True) #days last trans log


    plt.show()
    sns.boxplot(data=r, x='target', y=column, palette=["plum", "skyblue"])
    plt.show()

def remove_outliers(df):
    for i in df.columns:
        if i != 'target':
            if i !='random':
                column_values = df[i].values
                q25, q75 = percentile(column_values, 25), percentile(column_values, 75)
                iqr = q75 - q25
                cut_off = iqr * 10
                print(len(df),i)
                print(q25,q75)
                lower, upper = q25 - cut_off, q75 + cut_off

                df = df[(df[i]>= lower) & (df[i]<= upper)]

    return df

def remove_outliers2(upselling_model):
    %matplotlib inline


    initial = upselling_model.copy()
    upselling_model = upselling_model.reset_index()

    zeros = upselling_model[upselling_model['target'] == 0]
    ones = upselling_model[upselling_model['target'] == 1]
    zeros = zeros[[i for i in zeros.columns if i not in ('index', 'company_new', 'company', 'actual_plan','timestamp_ym', 'data_alta_basic', 'data_alta_premium', 'country', 'date_first_client','date_first_invoice','postal_code', 'company_name','xin_value', 'active_prem', 'target' ,'days_since_alta_premium','days_since_baixa_prem','days_until_premium','channel_sent','channel_rec')]]
    ones = ones[[i for i in ones.columns if i not in ('index', 'company_new', 'company', 'actual_plan','timestamp_ym', 'data_alta_basic', 'data_alta_premium', 'country', 'date_first_client','date_first_invoice','postal_code', 'company_name','xin_value', 'active_prem', 'target','days_since_alta_premium','days_since_baixa_prem','days_until_premium' ,'channel_sent','channel_rec')]]

    clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.15), \
                            max_features=1.0, bootstrap=False, n_jobs=-1, verbose=0)

    clf.fit(zeros)
    pred = clf.predict(zeros)
    zeros['anomaly']=pred
    outliers=zeros.loc[zeros['anomaly']==-1]
    outliers0_index=list(outliers.index)
    print(zeros['anomaly'].value_counts())



    final = initial.drop(initial.index[outliers0_index])#comentar aquestes dos si vull dibuixar els outliers
    #final = final.drop(initial.index[outliers1_index])


    return final

def standarize(upselling_model,test_companies):


    upselling_company = upselling_model.index
    upselling_target = upselling_model[['target']]
    upselling_X = upselling_model.iloc[:,:-1]
    upselling_X['random'] = np.random.randint(0,1000, size=len(upselling_X)) #random feature added
    upselling_X = upselling_X.reset_index()
    upselling_X = upselling_X.drop(columns=['company'])

    test_company= test_companies.index
    test_target = test_companies[['target']]
    test_X = test_companies.iloc[:,:-1]
    test_X['random'] = np.random.randint(0,1000, size=len(test_X)) #random feature added
    test_X = test_X.reset_index()
    test_X = test_X.drop(columns=['company'])

    std = StandardScaler()
    std.fit(upselling_X)

    scaled1 = std.transform(upselling_X)
    scaled2 = std.transform(test_X)

    scaled1 = pd.DataFrame(scaled1,columns=upselling_X.columns)
    scaled2 = pd.DataFrame(scaled2,columns=test_X.columns)

    upselling_X.rename(columns={column: column + "_original" for column in upselling_X.columns}, inplace=True)
    data1 = upselling_X.merge(scaled1, left_index = True, right_index=True, how = "left")
    data1['company'] = upselling_company
    data1 = data1.set_index('company')

    data1.drop(columns = [column for column in data1.columns if "_original" in column], inplace=True)
    upselling_model = data1.join(upselling_target)

    test_X.rename(columns={column: column + "_original" for column in test_X.columns}, inplace=True)
    data2 = test_X.merge(scaled2, left_index = True, right_index=True, how = "left")
    data2['company'] = test_company
    data2 = data2.set_index('company')
    data2.drop(columns = [column for column in data2.columns if "_original" in column], inplace=True)
    test_companies = data2.join(test_target)

    return upselling_model, test_companies


def plot_dist(churn, no_churn, attr):
    f,axes = plt.subplots(1,3,figsize=(25,10), sharex=True)
    ax1=sns.distplot(churn[attr], color="silver",ax=axes[0], bins=30, hist_kws=dict(edgecolor="grey",linewidth=2))
    ax2=sns.distplot(no_churn[attr], color="tan",ax=axes[1], bins=30, hist_kws=dict(edgecolor="peru",linewidth=2))
    ax3=sns.kdeplot(churn[attr], color="silver",ax=axes[2], legend=False)
    ax3=sns.kdeplot(no_churn[attr], color="tan",ax=axes[2], legend=False)
    ax3.set_xlabel(attr)
    sns.despine(top=True,right=True)
    f.text(0.07,0.5,'Frequency')
    f.suptitle(f'{attr}')
    f.legend(labels=['Churn','No Churn'])


def upsample(churn):
    churn_0 = churn[churn['target']==0]
    churn_1 = churn[churn['target']==1]

    churn_1_upsampled = resample(churn_1,
                                replace=True,     # sample with replacement
                                n_samples=len(churn_0),    # to match majority class
                                random_state=123)

    churn = pd.concat([churn_0, churn_1_upsampled])
    return churn

# %%
#READ STORED DATASETS
transactions_behaviour = pd.read_csv('transactions_behaviour_t.csv')
transactions_behaviour = transactions_behaviour.set_index('company')
transactions_behaviour = transactions_behaviour.drop(84326)

transactions_behaviour2 = pd.read_csv('transactions_behaviour.csv')
transactions_behaviour2 = transactions_behaviour2.set_index('company')
transactions_behaviour2 = transactions_behaviour2.drop(84326)

#%%
companies = pd.read_csv('companies.csv')
companies = companies.set_index('company')

churn = pd.read_csv('churn_t.csv')
churn = churn.set_index('company')

churn_b = pd.read_csv('churn_basic.csv')
churn_b = churn_b.set_index('company')

noww = pd.to_datetime(datetime.today(), utc=True)
transactions_behaviour2 = transactions_behaviour2.reset_index()
transactions_behaviour2['timestamp_ym'] = transactions_behaviour2['timestamp_ym'].astype('str')
transactions_behaviour2['company'] = transactions_behaviour2['company'].astype('str')
transactions_behaviour2['company_new'] = transactions_behaviour2['company']+'-'+transactions_behaviour2['timestamp_ym']
#%%
transactions_behaviour2 = transactions_behaviour2.set_index('company_new')

#%%
# CREATE DATASETS

companies['active_prem'] = companies['active_prem'].fillna(0)
upselling_1_ = companies[(companies['active_prem'] == 1)]
#si totes: upselling_1_ = companies[(companies['data_alta_premium'].isna()==True)]
upselling_1 = transactions_behaviour[transactions_behaviour.index.isin(upselling_1_.index)]
upselling_1 = upselling_1[(upselling_1['1']==0)]
upselling_1 = upselling_1[(upselling_1['days_since_alta_premium']<0) | (upselling_1['days_since_alta_premium'].isna()==True)]
upselling_1 = upselling_1.join(companies,how='left')
upselling_1 = upselling_1.reset_index()
upselling_1['timestamp_ym'] = upselling_1['timestamp_ym'].astype('str')
upselling_1['company'] = upselling_1['company'].astype('str')
upselling_1['company_new'] = upselling_1['company']+'-'+upselling_1['timestamp_ym']
upselling_1_ = upselling_1.groupby('company').tail(1)
upselling_1 = upselling_1.drop(upselling_1_.index)
upselling_1['target'] = 0
upselling_1_['target'] = 1
upselling_1 = upselling_1.append(upselling_1_)
upselling_1['company'] = upselling_1['company'].astype('float')
upselling_1 = upselling_1.set_index('company_new')
#%%
upselling_0_ = companies[(companies['active_prem'] == 0)]
upselling_0_ = upselling_0_[(~(upselling_0_.index.isin(churn_b.index)))]  #treure baixes basic
upselling_0 = transactions_behaviour[transactions_behaviour.index.isin(upselling_0_.index)]
upselling_0 = upselling_0[upselling_0['days_since_alta_premium'].isna() ==True]
upselling_0 = upselling_0.join(companies,how='left')
upselling_0 = upselling_0.reset_index()
upselling_0['timestamp_ym'] = upselling_0['timestamp_ym'].astype('str')
upselling_0['company'] = upselling_0['company'].astype('str')
upselling_0['company_new'] = upselling_0['company']+'-'+upselling_0['timestamp_ym']
upselling_0['company'] = upselling_0['company'].astype('float')
upselling_0 = upselling_0.set_index('company_new')
upselling_0['target'] = 0
#%%
upselling_model = upselling_1.append(upselling_0)

target = upselling_model[['target']]
upselling_model = upselling_model.drop(columns='target')

#%%
upselling_model = upselling_model.join(transactions_behaviour2[['acum_1', 'acum_0','sum_trans_sent']])
upselling_model = upselling_model.join(target)
#upselling_model.to_csv('upselling_dataset_temporal.csv')

#FINS AQUÍ CHAPTER 2
#%%
#visualitzacio del nans i tal
upselling_model['number_clients'] = upselling_model['number_clients'].fillna(0)
upselling_model['days_from_last_transaction'] = upselling_model['days_from_last_transaction'].fillna(upselling_model['days_since_alta_basic'])
upselling_model['days_from_last_transaction_rec'] = upselling_model['days_from_last_transaction_rec'].fillna(upselling_model['days_since_alta_basic'])
upselling_model['days_until_invoice'] = upselling_model['days_until_invoice'].fillna(upselling_model['days_since_alta_basic'])
upselling_model['days_until_client'] = upselling_model['days_until_client'].fillna(upselling_model['days_since_alta_basic'])
upselling_model[['trans_med_received','trans_med_sent', 'trans_max_received','trans_max_sent']] = upselling_model[['trans_med_received','trans_med_sent', 'trans_max_received','trans_max_sent']].fillna(0)

#upselling_model.to_csv('upselling_dataset_temporal.csv')

#upselling = upsample(upselling_0,upselling_1)


#%%
upselling_model = upselling_model.drop(columns=['company','actual_plan','timestamp_ym',
'data_alta_basic', 'data_alta_premium', 'country', 'date_first_client',
'date_first_invoice','postal_code', 'company_name','xin_value', 'active_prem',
'days_since_alta_premium','days_since_baixa_prem','days_until_premium','last_trans_plan','number_clients','acum_1','1','2'])

#%%
upselling_model.index.name = 'company'

test_companies = upselling_model.copy()
#%%
######################################################################## EDA
#Outliers
#descriptions(upselling_model, 'num_receivers')
#descriptions(upselling_model, 'med_trans_sent')
#%%
upselling_model = remove_outliers2(upselling_model)
#%%
#descriptions(upselling_model, 'med_trans_sent')
#descriptions(upselling_model, 'med_trans_sent',0,40)

#%%
#EDA univ
#descriptions(upselling_model, 'import_sum_sent',0,100000) #normal
#descriptions(upselling_model, 'import_sum_sent',0,10000000) #normal

#descriptions(upselling_model, 'acum_0', 0,125)
#descriptions(upselling_model, 'acum_0', 0,500000)

#descriptions(upselling_model, 'days_from_last_transaction', 0,1000)
#descriptions(upselling_model, 'days_from_last_transaction',0,1000)
#%%
#EDA multi
#corr_matrix(upselling_model)
#joint_plots(['trans_sum_sent','days_until_invoice','import_sum_sent','days_from_last_transaction','num_receivers','sum_trans_sent','days_since_alta_basic','max_import_sent','med_import_sent_t','target'], upselling_model)
#%%
upselling_model, test_companies = standarize(upselling_model, test_companies)
#%%
#joint_plots(['num_receivers','sum_trans_sent','target'], upselling_model, 50)
#%%
#joint_plots(['days_until_invoice','days_until_client','target'], upselling_model, 10000)
#%%
#joint_plots(['0','trans_sum_sent','success_true','target'], upselling_model,10000)
#%%
#joint_plots(['days_from_last_transaction','med_trans_sent_m','target'], upselling_model,10000)




upselling_model = upselling_model.drop(columns=['acum_0','0','success_true','trans_max_sent','last_trans_plan_rec','b2g_acum','other_acum','mail_acum','downloads_acum','b2brouter_acum','peppol_acum','peppol','mail','b2brouter','downloads'])
#%%
train_x, val_x, train_y, val_y, cols, companies_test = separate_predict(upselling_model, test_companies , oversample=False)

# %%
##############################################################################################################
#


#DEIFNITIU
param_grid = {
    "eval_metric" : ['logloss'],
    "learning_rate" : [0.01],
    "n_estimators" : [110], #150
    "colsample_bytree":[0.7], #1
    "scale_pos_weight" : [3],
    "subsample":[0.6],
    "max_depth" : [3], #2
    "reg_lambda": [1],
    "min_child_weight":[1],
    "gamma":[15]
}


'''
#GRIDSEARCH
param_grid = {
    "eval_metric" : ['logloss'],
    "learning_rate" : [0.05],
    "n_estimators" : [100],
    "colsample_bytree":[0.7],
    "scale_pos_weight" : [3],
    "subsample":[0.7],
    "max_depth" : [4],
    "reg_lambda": [0.8],
    "min_child_weight":[3],
}'''


model = XGBClassifier()
grid_cv = find_best_params(train_x, val_x, train_y, val_y,model,param_grid)
final_model = XGBClassifier(**grid_cv.best_params_)
classification, probabilities,predictions =  train_model_and_performance_eval(final_model, train_x, val_x, train_y, val_y, cols, 0.5,'features')
#%%
false_pos = list((test_companies.iloc[list(classification[1].index)]).index)
not_tested = list(test_companies.index)



probabilities_dict = {}

#%%
while len(not_tested) > 0:
    train_x, val_x, train_y, val_y, cols, companies_test = separate_predict(upselling_model,test_companies , oversample=False)
    classification, probabilities, predictions  = train_model_and_performance_eval(final_model, train_x, val_x, train_y, val_y, cols, 0.5,'features')
    for idd in companies_test:
        if idd in probabilities_dict:
            pass
        else:
            probabilities_dict[idd] = probabilities[list(companies_test).index(idd)]
    false_pos = list(set(false_pos).union(list( (test_companies.iloc[list(classification[1].index)]).index )))
    not_tested = [ele for ele in not_tested if ele not in list(companies_test)]

# %%
def prepare_dict(dict):
    secondElements = {}
    for key, value in dict.items():
        secondElements[key] = value[1]
    return secondElements


d = prepare_dict(probabilities_dict)
a = dict(sorted(d.items(), key=lambda item: item[1], reverse=True)) #all companies with prob if we take all its data before becoming premium
current0 = test_companies[test_companies['target']==0]
final_dictionary = {k: v for k, v in a.items() if k in current0.index} #these are only the ones that are currently 0

#%%
def plot_evolution_hyperparam(n_estimators):
    %matplotlib inline
    train_results = []
    test_results = []
    for estimator in n_estimators:
        rf = XGBClassifier(eval_metric='logloss', n_estimators=110, learning_rate=0.01, colsample_bytree=estimator)#, colsample_bytree=estimator)
        rf.fit(train_x, train_y)
        train_pred = rf.predict(train_x)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(val_x)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    line1, = plt.plot(n_estimators, train_results, label='Train AUC', color='lightskyblue')
    line2, = plt.plot(n_estimators, test_results,  label='Test AUC', color='purple')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('colsample_bytree')
    plt.show()
    return
#e = [0.005, 0.01,0.05,0.1,0.2]
e = [100,105,110,115,120]
e = [0,0.2,0.4,0.5,0.6,0.8,1]
#e = [1,2,3,4,5,6,7,8,9,10,15]
plot_evolution_hyperparam(e)
# %%
