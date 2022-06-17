#%%
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from collections import Counter

from datetime import date
import warnings
import eland as ed
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from decouple import config
from sqlalchemy import column
from elastic import get_elastic

warnings.filterwarnings("ignore")
es = get_elastic()
now = pd.to_datetime(datetime.today())
pd.set_option('display.max_columns', None)

#%%
def drop_columns_ba(df):
    df = df.drop(columns=['0', '1', '2', 'days_since_alta_premium'
                        'success_true', 'success_false',
                        'number_trans_received', 'number_trans_sent']) #'Catalunya','Espanya B2G', 'França', 'Italia', 'Peppol', 'EDI', 'Mail','Altres xarxes', 'Downloads', 'B2B Espanya', 'Canals B2B privats','B2Brouter', 'Andorra', 'Portugal', 'Inditex', 'international'
    return df

def drop_columns_pre(df):
    df = df.drop(columns=[]) #'Catalunya','Espanya B2G', 'França', 'Italia', 'Peppol', 'EDI', 'Mail','Altres xarxes', 'Downloads', 'B2B Espanya', 'Canals B2B privats','B2Brouter', 'Andorra', 'Portugal', 'Inditex', 'international'
    return df

def median_calculation(transactions_behaviour):

    last_rows = transactions_behaviour.groupby('company').tail(1)

    last_rows = last_rows.drop(columns=['0', '1', '2','success_true', 'success_false','trans_sum_received','trans_sum_sent',
                                        'import_med_rec', 'import_max_rec','import_sum_rec','import_med_sent','import_max_sent','import_sum_sent'])


    return last_rows

def company_info():
    company = ed.DataFrame(
        es_client=es,
        es_index_pattern='company*')
    company = company[company['plan']!='enterprise']
    company = company[company['company_id']>10] #
    company = company[['plan','postal_code','date_last_updated','company_id']] #
    company = ed.eland_to_pandas(company)
    company = company.sort_values(by='date_last_updated')
    company = company.drop_duplicates(subset='company_id',keep='last')

    timestamp_ym = pd.to_datetime(company['date_last_updated']).dt.strftime('%Y%m')
    company['date_last_updated_ym'] = timestamp_ym
    #company = company[company['date_last_updated']<'2022-06-15 12:21:00+00:00']

    return company
#%%
transactions_behaviour = pd.read_csv('transactions_behaviour_rfm3.csv')
transactions_behaviour = transactions_behaviour.set_index('company')
#%%
companies = pd.read_csv('companies.csv')
companies = companies.set_index('company')

churn = pd.read_csv('churn.csv')
churn = churn.set_index('company')

churn_b = pd.read_csv('churn_basic.csv')
churn_b = churn_b.set_index('company')
#%%
noww = pd.to_datetime(datetime.today(), utc=True)
company = company_info()
company = company.set_index('company_id')
company.index.name = 'company'

#company = company[['data_alta', 'data_subscripcio']]

#%%
transactions_behaviour = transactions_behaviour[(~(transactions_behaviour.index.isin(churn_b.index)) )]  #treure baixes basic
#%%

#%%
last_rows = median_calculation(transactions_behaviour)
#%%
#companies = companies.drop(columns=['channel_sent','channel_rec'])
last_rows = last_rows.join(companies,how='left') #info about company
last_rows['days_between_altas'] = last_rows['days_since_alta_basic'] - last_rows['days_since_alta_premium']
#%%
dataset = last_rows
#%%
dataset = dataset[((~(dataset.index.isin(churn.index))) | ( (dataset.index.isin(churn.index)) &  (dataset['days_from_last_transaction'] < dataset['days_since_baixa_prem']) ))]  #treure baixes basic

#agafar nomes acumulades
#%%
dataset['channel'] = ['']*len(dataset)

dataset['channel'] = dataset[['b2g_acum', 'mail_acum',
       'downloads_acum', 'other_acum', 'peppol_acum', 'b2brouter_acum']].idxmax(axis=1)
dataset['check'] = dataset[['b2g_acum', 'mail_acum',
       'downloads_acum', 'other_acum', 'peppol_acum', 'b2brouter_acum']].sum(axis=1)

for i in dataset.index:
    if dataset['check'][i] == 0:
        dataset['channel'][i]='no channel'
    else:
        pass
#%%

# if basic ###############
dataset = dataset[['med_import_sent_t','days_since_alta_basic','days_from_last_transaction','company_name',
                    'country','data_alta_basic','data_alta_premium','med_trans_sent','sum_trans_sent','num_receivers','num_senders','days_since_baixa_prem','days_until_premium', 'days_until_invoice', 'channel']]
dataset.columns = ['import','days_since_alta_basic','days_until_last_transaction','company_name',
                    'country','data_alta_basic','subscription_date','num_transactions','sum_transactions','receivers','senders','days_since_baixa_prem','days_until_premium', 'days_until_invoice','channel']
#%%
dropss = dataset[~dataset.index.isin(company.index)]
#%%
dataset = dataset.drop(dropss.index)
dataset = dataset.join(company[['plan']])
#%%
dataset = dataset.join(company[['postal_code']])
'''prova2.columns = ['trans_lastyear']
dataset = dataset.join(prova2)'''

#%%
def rfm(b_1,b_2,data):
    data['r']=''
    data['f']=''
    data['m']=''

    for i in range(len(data)):
        if data.index[i]!=0.0:
            if data['import'][data.index[i]] <=b_1[0]:
                data['m'][data.index[i]] = 1
            if  b_1[0] < data['import'][data.index[i]] <= b_1[1]:
                data['m'][data.index[i]] = 2
            if b_1[1] < data['import'][data.index[i]] <= b_1[2]:
                data['m'][data.index[i]] = 3
            if data['import'][data.index[i]] > b_1[2]:
                data['m'][data.index[i]] = 4

            if data['num_transactions'][data.index[i]] ==b_2[0]:
                data['f'][data.index[i]] = 1
            if b_2[0] < data['num_transactions'][data.index[i]] <= b_2[1] :
                data['f'][data.index[i]] = 2
            if b_2[1] < data['num_transactions'][data.index[i]] <= b_2[2] :
                data['f'][data.index[i]] = 3
            if data['num_transactions'][data.index[i]] > b_2[2]:
                data['f'][data.index[i]] = 4
            try:
                if int(data['days_until_last_transaction'][data.index[i]]) <= 30:
                    data['r'][data.index[i]] = 4
            except:
                if int(data['days_since_alta_basic'][data.index[i]]) <= 30:
                    data['r'][data.index[i]] = 4
                    continue
                if 30 < int(data['days_since_alta_basic'][data.index[i]]) <= 95:
                    data['r'][data.index[i]] = 3
                    continue
                if 95 < int(data['days_since_alta_basic'][data.index[i]]) <= 365:
                    data['r'][data.index[i]] = 2
                    continue
                if int(data['days_since_alta_basic'][data.index[i]]) > 365:
                    data['r'][data.index[i]] = 1
                    continue
            if 30 < int(data['days_until_last_transaction'][data.index[i]]) <= 95:
                data['r'][data.index[i]] = 3
            if 95 < int(data['days_until_last_transaction'][data.index[i]]) <= 365:
                data['r'][data.index[i]] = 2
            if int(data['days_until_last_transaction'][data.index[i]]) > 365:
                data['r'][data.index[i]] = 1

    return data

# prepare dataset rfm
def prepare_dataset_rfm(data):

    rfm_dataset= rfm(buckets_money, buckets_freq,data)
    rfm_dataset['company_id']=rfm_dataset.index
    return rfm_dataset
#%%
def upload_dataset(df, name):

    df['ingest_date'] = datetime.today()
    df['ingest_date'] = pd.to_datetime(df['ingest_date'])
    df['subscription_date'] = pd.to_datetime(df['subscription_date'], utc=True)
    df = df.fillna(0.0)
    df['r'] = df['r'].astype('int')
    # concat name + today string
    #name = name + (str(datetime.today())[0:4])+(str(datetime.today())[5:7])
    return df

#%%
dataset['plan'] = dataset['plan'].fillna('basic')
#%%
buckets_money=[200,1100,5000]
buckets_freq=[0,1,4]
rfm_prem = prepare_dataset_rfm(dataset[(dataset['plan']=='professional') | (dataset['plan']=='business')])

#%%
buckets_money=[0,5000,50000]
buckets_freq=[0,1,2]
rfm_basic = prepare_dataset_rfm(dataset[dataset['plan']=='basic'])
# %%

rfm_premium = rfm_prem.copy()
#%%
#%%
rfm_basic['segment'] = ''
#%%
rfm_premium['segment'] = ''
#%%
#rfm_enterprise['segment'] = ''
for i in range(len(rfm_basic)):
    if rfm_basic['days_until_last_transaction'][i] < rfm_basic['days_since_baixa_prem'][i] :
        rfm_basic['segment'][rfm_basic.index[i]] = 'Downgrades'
        continue

    if ((rfm_basic['r'][rfm_basic.index[i]] >= 3) & (int(rfm_basic['sum_transactions'][rfm_basic.index[i]]) == 0)):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Recent clients'
        continue

    if ((rfm_basic['r'][rfm_basic.index[i]] == 1) & (int(rfm_basic['sum_transactions'][rfm_basic.index[i]]) > 0)):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Lost'
        continue

    if ((rfm_basic['r'][rfm_basic.index[i]] < 3) & (int(rfm_basic['sum_transactions'][rfm_basic.index[i]]) == 0)):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Inactive'
        continue

    if (rfm_basic['receivers'][rfm_basic.index[i]] ==1):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Low engagement'
        continue


    if ((rfm_basic['r'][rfm_basic.index[i]] >=2) & (rfm_basic['f'][rfm_basic.index[i]] >=3)): # 1 o 2
        rfm_basic['segment'][rfm_basic.index[i]] = 'Normal'
        continue

    if ((rfm_basic['r'][rfm_basic.index[i]] >=3) & (rfm_basic['f'][rfm_basic.index[i]] ==2) & (rfm_basic['m'][rfm_basic.index[i]] !=4)):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Normal'
        continue

    if ((rfm_basic['r'][rfm_basic.index[i]] >= 3 ) & (rfm_basic['f'][rfm_basic.index[i]] >=2 ) & (rfm_basic['m'][rfm_basic.index[i]] == 4)):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Bullies'
        continue

    if ((rfm_basic['r'][rfm_basic.index[i]] >=4 ) & (rfm_basic['f'][rfm_basic.index[i]] >= 3) & (rfm_basic['m'][rfm_basic.index[i]] == 3)):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Potential premiums'
        continue

    if ((rfm_basic['r'][rfm_basic.index[i]] == 2) & (rfm_basic['f'][rfm_basic.index[i]] ==2 )):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Sporadic'
        continue

    if ((rfm_basic['r'][rfm_basic.index[i]] >= 2) & (rfm_basic['f'][rfm_basic.index[i]] ==1 )):
        rfm_basic['segment'][rfm_basic.index[i]] = 'Sporadic'
        continue
#%%
#segmentation premium
for i in range(len(rfm_premium)):
    if rfm_premium['days_until_last_transaction'][i] < rfm_premium['days_since_baixa_prem'][i] :
        rfm_premium['segment'][rfm_premium.index[i]] = 'Downgrades'
        continue
    if ((rfm_premium['r'][rfm_premium.index[i]] >= 3) & (int(rfm_premium['sum_transactions'][rfm_premium.index[i]]) == 0)):
        rfm_premium['segment'][rfm_premium.index[i]] = 'Recent clients'
        continue

    if ((rfm_premium['r'][rfm_premium.index[i]] <=2)):
        rfm_premium['segment'][rfm_premium.index[i]] = 'Almost lost'
        continue

    if (rfm_premium['receivers'][rfm_premium.index[i]]==1):
        rfm_premium['segment'][rfm_premium.index[i]] = 'Low engagement'
        continue

    if ((rfm_premium['r'][rfm_premium.index[i]] >= 3) & (rfm_premium['f'][rfm_premium.index[i]] >= 2) & (rfm_premium['m'][rfm_premium.index[i]] == 4)):
        rfm_premium['segment'][rfm_premium.index[i]] = 'Big companies'
        continue

    if ((rfm_premium['r'][rfm_premium.index[i]] >= 3) & (rfm_premium['f'][rfm_premium.index[i]] >= 3) & (rfm_premium['m'][rfm_premium.index[i]] <4)):
        rfm_premium['segment'][rfm_premium.index[i]] = 'Loyals'
        continue

    if ((rfm_premium['r'][rfm_premium.index[i]] >=3) & (rfm_premium['f'][rfm_premium.index[i]] == 2) &  (rfm_premium['m'][rfm_premium.index[i]] < 4)):
        rfm_premium['segment'][rfm_premium.index[i]] = 'Normal'
        continue

    if ((rfm_premium['r'][rfm_premium.index[i]] >=3) & (rfm_premium['f'][rfm_premium.index[i]] == 1)):
        rfm_premium['segment'][rfm_premium.index[i]] = 'Sporadic'
        continue


#%%
rfm_premium = rfm_premium.drop(columns=['days_since_baixa_prem'])
#%%
rfm_premium['days_until_last_transaction'] = rfm_premium['days_until_last_transaction'].fillna(rfm_premium['days_since_alta_basic'])
rfm_premium['days_until_invoice'] = rfm_premium['days_until_invoice'].fillna(rfm_premium['days_since_alta_basic'])
rfm_premium['subscription_date'] = rfm_premium['subscription_date'].fillna('1900-12-02T10:59:33+00:00')
rfm_premium['subscription_date'] = pd.to_datetime(rfm_premium['subscription_date'])
rfm_premium['data_alta_basic'] = pd.to_datetime(rfm_premium['data_alta_basic'])

#%%
rfm_basic['days_until_last_transaction'] = rfm_basic['days_until_last_transaction'].fillna(rfm_basic['days_since_alta_basic'])
rfm_basic['days_until_invoice'] = rfm_basic['days_until_invoice'].fillna(rfm_basic['days_since_alta_basic'])
rfm_basic['days_until_premium'] = rfm_basic['days_until_premium'].fillna(rfm_basic['days_since_alta_basic'])
rfm_basic['data_alta_basic'] = pd.to_datetime(rfm_basic['data_alta_basic'])

#%%
rfm_basic = rfm_basic.drop(columns=[ 'days_since_baixa_prem','subscription_date'])

#%%
rfm_premium = rfm_premium.dropna()
#%%
rfm_basic = rfm_basic.dropna()
rfm_premium.to_csv('rfm_premium2.csv')
rfm_basic.to_csv('rfm_basic2.csv')
#%%
rfmm = rfm_premium.append(rfm_basic)

rfmm.to_csv('segmentation_model.csv')
