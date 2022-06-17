#%%
from datetime import datetime
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
from numpy import nan
import random
import string
pd.set_option('display.max_columns', None)
#%%
rfm=True
warnings.filterwarnings("ignore")
es = get_elastic()
now = pd.to_datetime(datetime.today())
TODAY_MONTH = str(datetime.today().year) + str(datetime.today().month).zfill(2)


#%%
# useful functions
# Returns de difference between two dates
def processNan (x):
    return 'IN'+''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(9))

def difference_dates(end,start):
    m_1=int(str(end).split("-")[1])
    m_2=int(str(start).split("-")[1])
    d_1=int(((str(end).split("-"))[2]).split(" ")[0])
    d_2=int(((str(start).split("-"))[2]).split(" ")[0])
    y_1=int(str(end).split("-")[0])
    y_2=int(str(start).split("-")[0])
    f_date = date(y_2, m_2, d_2)
    l_date = date(y_1, m_1, d_1)
    delta = l_date - f_date
    difference=delta.days
    return difference

def subtract_days(startdate, leftdays):
    enddate = pd.to_datetime(startdate) - pd.DateOffset(days=leftdays)
    return enddate

def subtract_months(startdate, leftdays):
    enddate = pd.to_datetime(startdate) - pd.DateOffset(months=leftdays)
    return enddate

def replace_plans(df):

    df = df.replace({'basic': 0})
    df = df.replace({'professional': 1})
    df = df.replace({'business': 2})
    df = df.replace({'enterprise': 3})
    return df

def transactions_info():
    transactions = ed.DataFrame(
        es_client=es,
        es_index_pattern='transactions-*')

    transactions = transactions[(transactions['company_plan']!='enterprise') & (transactions['company_plan']!='aplicateca2') & (transactions['company_plan']!='aplicateca5')]
    transactions = transactions[(transactions['company_id']>10)]
    transactions = transactions[['timestamp','id','company_id','company_plan','channel','success', 'state','sender_name',
                                'sender_country','sendable_type', 'import','international','kind','document_kind','receiver_name','sender_xin_value', 'receiver_xin_value']] #receiver o sender?
    transactions = transactions[transactions['sender_xin_value']!='#personal xin']
    transactions = transactions[transactions['receiver_xin_value']!='#personal xin']
    block = int(len(transactions)/200)+1
    rangemax = block
    rangemin = 0
    maxid = transactions['id'].max()
    for i in range(1000):
        if rangemin >= maxid:
            break
        chunk = transactions[(transactions['id']>=rangemin) & (transactions['id']<=rangemax)]
        chunk = ed.eland_to_pandas(chunk)
        rangemax += block
        rangemin += block
        if i==0:
            dataset=chunk
        else:
            frames = [dataset, chunk]
            dataset = pd.concat(frames)

    timestamp_ym = pd.to_datetime(dataset['timestamp']).dt.strftime('%Y%m')
    dataset['timestamp_ym'] = timestamp_ym
    return dataset

def company_info():
    company = ed.DataFrame(
        es_client=es,
        es_index_pattern='company*')
    company = company[company['plan']!='enterprise']
    company = company[company['company_id']>10] #assegurar que funciona, sinó fer loop
    company = company[['company_id','country','data_alta','data_subscripcio', 'plan','date_first_client','date_first_invoice',
                        'number_clients', 'postal_code','date_last_updated','company_name', 'xin_value']] ###company_name
    company = ed.eland_to_pandas(company)
    company = company.sort_values(by='date_last_updated')
    company = company.drop_duplicates(subset='company_id',keep='last')

    timestamp_ym = pd.to_datetime(company['date_last_updated']).dt.strftime('%Y%m')
    company['date_last_updated_ym'] = timestamp_ym
    return company

def revenue_info():
    revenue = ed.DataFrame(
        es_client=es,
        es_index_pattern='revenue*')

    revenue = revenue[(revenue['company']>10) & (revenue['company']!=504)] #assegurar que funciona, sinó fer loop
    revenue = ed.eland_to_pandas(revenue)

    #1st weird case
    drop_ex1 = (revenue[(revenue['action']=='renovacio') & (revenue['company']==150702) & (revenue['altes']==110)]).index
    revenue = revenue.drop(drop_ex1)

    #2nd weird case
    drop_ex2 = (revenue[(revenue['action']=='alta') & (revenue['company']==123419) & (revenue['plan']=='basic')])
    drop_ex2_time = drop_ex2['@timestamp'][drop_ex2.index[0]]
    drop_ex3 = (revenue[(revenue['action']=='alta') & (revenue['company']==123419) & (revenue['plan']=='professional')])
    drop_ex3_time = drop_ex3['@timestamp'][drop_ex3.index[0]]
    revenue = revenue.drop(drop_ex2.index)
    revenue = revenue.drop(drop_ex3.index)
    drop_ex2['@timestamp'][drop_ex2.index[0]]=drop_ex3_time
    drop_ex3['@timestamp'][drop_ex3.index[0]]=drop_ex2_time
    revenue = revenue.append(drop_ex2)
    revenue = revenue.append(drop_ex3)

    #2nd weird case
    drop_ex2 = (revenue[(revenue['action']=='alta') & (revenue['company']==100453) & (revenue['plan']=='basic')])
    drop_ex2_time = drop_ex2['@timestamp'][drop_ex2.index[0]]
    drop_ex3 = (revenue[(revenue['action']=='alta') & (revenue['company']==100453) & (revenue['plan']=='professional')])
    drop_ex3_time = drop_ex3['@timestamp'][drop_ex3.index[0]]
    revenue = revenue.drop(drop_ex2.index)
    revenue = revenue.drop(drop_ex3.index)
    drop_ex2['@timestamp'][drop_ex2.index[0]]=drop_ex3_time
    drop_ex3['@timestamp'][drop_ex3.index[0]]=drop_ex2_time
    revenue = revenue.append(drop_ex2)
    revenue = revenue.append(drop_ex3)

    revenue = revenue.drop_duplicates()
    revenue = revenue[(revenue['action']!='autorenovacio on') & (revenue['action']!='autorenovacio off')]

    revenue = revenue.sort_values(['@timestamp', 'date_last_updated'], ascending=[True,True])

    drop = revenue.groupby(['company']).agg({'plan':'last','action':'last'})
    drop = (drop[(drop['plan']=='enterprise') | (drop['plan']=='aplicateca2') | (drop['plan']=='aplicateca5') ] ).index

    drop = (revenue[revenue['company'].isin(drop)]).index
    revenue = revenue.drop(drop)
    revenue = revenue[revenue['company_name']!='Pruebas PRE 917080021']
    revenue = revenue[(revenue['plan']!='aplicateca2') & (revenue['plan']!='aplicateca5')]
    timestamp_ym = pd.to_datetime(revenue['@timestamp']).dt.strftime('%Y%m')
    revenue['timestamp_ym'] = timestamp_ym

    return revenue

def churn_basic(revenue):
    churn = revenue[(revenue['plan']==0)]
    churn = churn.groupby(['company']).agg({'@timestamp': 'last','action':'last','plan':'last'})
    churn = churn[(churn['action']=='baixa')]
    return churn


def churn_dataset(revenue):
    '''
    PREMIUM
    '''
    churn = revenue[revenue['payment_type']!='FreeDemo']
    churn = churn[(churn['plan']==1) | (churn['plan']==2)]
    churn = churn.groupby(['company']).agg({'@timestamp': 'last','action':'last','plan':'last'})
    churn = churn[(churn['action']=='baixa') | (churn['action']=='downgrade') | (churn['action']=='devolucio')]
    churn['data_baixa_prem'] = pd.to_datetime(churn['@timestamp'], utc=True)
    churners = churn.drop(columns=['action', '@timestamp'])

    alta = revenue[(revenue['action']=='alta')]
    alta = alta[(alta['plan']==1) | (alta['plan']==2)]
    alta = alta.groupby('company').agg({'@timestamp':'first'})
    alta['data_alta_prem'] = pd.to_datetime(alta['@timestamp'], utc=True)
    alta = alta.drop(columns=['@timestamp'])

    #churners = pd.concat([churners, alta],axis=1,join='inner')
    churners = churners.join(alta, how ='left')

    churners['days_active_prem'] = (churners['data_baixa_prem'] - churners['data_alta_prem']).dt.days
    churners['churn_plan'] = churners['plan']
    churners = churners.drop(columns=['data_alta_prem','plan'])
    churners['churn'] = 1
    churners.index.name = 'company'
    return churners


def all_users(revenue,company):

    alta = revenue[(revenue['action']=='alta')]
    alta = alta.groupby('company').agg({'@timestamp':'first'})
    alta['data_alta_basic'] = pd.to_datetime(alta['@timestamp'],utc=True)
    alta = alta.drop(columns=['@timestamp'])

    alta_p = revenue[(revenue['plan']==1) | (revenue['plan']==2)]
    alta_p = alta_p[(alta_p['action']=='alta') | (alta_p['action']=='upgrade')]
    alta_p = alta_p.groupby('company').agg({'@timestamp':'first'})
    alta_p['data_alta_premium'] = pd.to_datetime(alta_p['@timestamp'],utc=True)
    alta_p = alta_p.drop(columns=['@timestamp'])
    alta = pd.concat([alta,alta_p],axis=1)

    company_dataset = company.drop(columns=['plan', 'data_subscripcio', 'data_alta','date_last_updated','date_last_updated_ym'])
    company_dataset = company_dataset.set_index('company_id')
    company_information= alta.join(company_dataset)

    company_information['days_until_premium'] = (company_information['data_alta_premium'] - company_information['data_alta_basic']).dt.days
    company_information['days_until_invoice'] = (company_information['date_first_invoice'] - company_information['data_alta_basic']).dt.days
    company_information['days_until_client'] = (company_information['date_first_client'] - company_information['data_alta_basic']).dt.days

    return company_information



def time_since_alta(dataset,datee, users, churn):
    dataset['days_since_alta_basic'] = [np.nan] * len(dataset)
    dataset['days_since_alta_premium'] = [np.nan] * len(dataset)
    dataset['days_since_baixa_prem'] = [np.nan] * len(dataset)

    datee = pd.to_datetime(datee, utc=True)
    for i in dataset.index:
        dataset['days_since_alta_basic'][i] = (datee - users['data_alta_basic'][i]).days

        dataset['days_since_alta_premium'][i] = (datee - users['data_alta_premium'][i]).days
        try:
            dataset['days_since_baixa_prem'][i] = (datee - churn['data_baixa_prem'][i]).days
        except:
            pass

    return dataset

def last_plan(revenue):
    alta = revenue.groupby('company').agg({'action':'last', 'plan':'last'})
    alta['actual_plan'] = alta['plan']
    alta = alta.drop(columns=['action','plan'])
    return alta

def active_prem(revenue):
    alta = revenue.groupby('company').agg({'action':'last', 'plan':'last','payment_type':'last'})

    alta = alta[(alta['action'] == 'alta') | (alta['action'] == 'upgrade') | (alta['action']=='renovacio') | (alta['action']=='reactivacio')]
    alta = alta[(alta['plan'] == 1) | (alta['plan']==2)]
    alta = alta[alta['payment_type'] != 'FreeDemo']

    #alta_p['data_last_subscription'] = pd.to_datetime(alta_p['@timestamp'],utc=True)
    alta['active_prem'] = 1
    alta = alta.drop(columns=['action','plan','payment_type'])
    return alta

def actives(revenue):
    alta = revenue.groupby('company').agg({'action':'last', 'plan':'last'})
    #alta_p['data_last_subscription'] = pd.to_datetime(alta_p['@timestamp'],utc=True)
    return alta

def transactions_byplan(trans):

    for i in range(3):
        df1 = trans.groupby('receiver_xin_value')['company_plan'].apply(lambda x: (x==i).sum()).reset_index(name=i).set_index('receiver_xin_value')
        if i ==0:
            start1 = df1
            continue
        start1 = pd.concat([start1, df1], axis=1)
    start1.index.name = 'xin_value'

    #start1.index.name = 'company'

    for i in range(3):
        df2 = trans.groupby('sender_xin_value')['company_plan'].apply(lambda x: (x==i).sum()).reset_index(name=i).set_index('sender_xin_value')
        if i ==0:
            start2 = df2
            continue
        start2 = pd.concat([start2, df2], axis=1)
    start2.index.name = 'xin_value'


    start = start1.append(start2)
    start.index.name = 'xin_value'
    start = start.groupby('xin_value').sum()

    return start

def last_transaction_plan(trans):
    plans1 = trans.groupby(['sender_xin_value']).agg({'company_plan':'last'})
    plans1['last_trans_plan'] = plans1['company_plan']
    plans1 = plans1.drop(columns='company_plan')
    plans1.index.name = 'xin_value'


    plans2 = trans.groupby(['receiver_xin_value']).agg({'company_plan':'last'})
    plans2['last_trans_plan_rec'] = plans2['company_plan']
    plans2 = plans2.drop(columns='company_plan')
    plans2.index.name = 'xin_value'

    plans = pd.concat([plans1, plans2], axis=1)

    plans['last_trans_plan'] = plans['last_trans_plan'].fillna(-1)
    plans['last_trans_plan'] = plans['last_trans_plan'].astype('int')

    return plans


def num_rec_sent(trans):
    sent = trans[(trans['state']=='sent') | (trans['kind']=='issue')]
    recieve = trans[(trans['state']=='received') | (trans['kind']=='reception')]

    recfrom = recieve.groupby('receiver_xin_value').agg({"sender_xin_value": pd.Series.nunique}) ###
    recfrom.index.name = 'xin_value'
    recfrom.columns = ['num_senders1']
    real_companies = list(recfrom.index)

    sento_ = sent.groupby('receiver_xin_value').agg({"sender_xin_value": pd.Series.nunique}) ###
    sento_.index.name = 'xin_value'
    sento_.columns = ['num_senders2']

    recfrom = pd.concat([recfrom,sento_], axis = 1)
    recfrom = recfrom.fillna(0)
    recfrom['num_senders'] = recfrom['num_senders1'] + recfrom['num_senders2']
    recfrom = recfrom.drop(columns=['num_senders1','num_senders2'])

    sento = sent.groupby('sender_xin_value').agg({"receiver_xin_value": pd.Series.nunique}) ###
    sento.index.name = 'xin_value'
    sento.columns = ['num_receivers1']
    real_companies.append(list(sento.index))
    real_companies =[item for sublist in real_companies for item in sublist]


    recfrom_ = recieve.groupby('sender_xin_value').agg({"receiver_xin_value": pd.Series.nunique}) ###
    recfrom_.index.name = 'xin_value'
    recfrom_.columns = ['num_receivers2']

    sento = pd.concat([sento,recfrom_], axis = 1)
    sento = sento.fillna(0)
    sento['num_receivers'] = sento['num_receivers1'] + sento['num_receivers2']
    sento = sento.drop(columns=['num_receivers1','num_receivers2'])


    num_people = pd.concat([sento,recfrom], axis = 1 )
    num_people = num_people[num_people.index.isin(real_companies)]
    num_people = num_people.fillna(0)

    return num_people

def trans_success(trans):
    trans_true = trans[(trans['success']==True)  | ((trans['success']!=False) & (trans['state']!='refused') & (trans['state']!='error') & (trans['state']!='discarded') & (trans['state']!='cancelled') & (trans['state']!='ocr_failed') & (trans['state']!='quote_expired') & (trans['state']!='quote_error') & (trans['state']!='quote_refused'))]
    trans_false = trans[(trans['success']==False) | ((trans['success']!=True) & (trans['state']=='refused') | (trans['state']=='error') | (trans['state']=='discarded') | (trans['state']=='cancelled') | (trans['state']=='ocr_failed') | (trans['state']=='quote_expired') | (trans['state']=='quote_error') | (trans['state']=='quote_refused'))]

    success_true_s = trans_true.groupby('sender_xin_value').agg({'id':'count'}) ##
    success_true_s.index.name = 'xin_value'
    success_true_s.columns = ['success_true_s']

    success_true_r = trans_true.groupby('receiver_xin_value').agg({'id':'count'}) ##
    success_true_r.index.name = 'xin_value'
    success_true_r.columns = ['success_true_r']

    success_true = pd.concat([success_true_r,success_true_s], axis = 1 )
    success_true = success_true.fillna(0)
    success_true['success_true'] = success_true['success_true_s']+success_true['success_true_r']
    success_true = success_true.drop(columns=['success_true_s','success_true_r'])

    success_false_s = trans_false.groupby('sender_xin_value').agg({'id':'count'})
    success_false_s.index.name = 'xin_value'
    success_false_s.columns = ['success_false_s']

    success_false_r = trans_false.groupby('receiver_xin_value').agg({'id':'count'})
    success_false_r.index.name = 'xin_value'
    success_false_r.columns = ['success_false_r']

    success_false = pd.concat([success_false_r,success_false_s], axis = 1 )
    success_false = success_false.fillna(0)
    success_false['success_false'] = success_false['success_false_s']+success_false['success_false_r']
    success_false = success_false.drop(columns=['success_false_s','success_false_r'])



    success = pd.concat([success_true, success_false], axis=1)
    success = success.fillna(0)
    return success

def channel_replacement(trans):
    trans['international'].replace({False: 0, True: 1}, inplace=True)

    #channels
    channels = ['Catalunya','Espanya B2G','França','Italia','Peppol','EDI','Mail',
                'Altres xarxes','Downloads','B2B Espanya','Canals B2B privats', 'B2Brouter',
                'Andorra','Portugal', 'Inditex']


    # B2G
    B2G = ['facturae_32_cabb', 'osakidetza', 'facturae_32_xunta', 'facturae_32_junta_andalucia', 'facturae_32_efaktur', 'facturae_32_face','facturae_32_face_detached',
     'biscaytik', 'dfbizkaia', 'dfalava', 'facturae_32_canaries','facturae_32_jccm','gipuzkoa','facturae_32_vitoria','facturae_32_govern_basc', 'es_publico', 'aoc32',
     'facturae_32_rioja','facturae_32_junta_an','facturae_32','facturae_32_jcyl','link_to_facturae32_b','facturae_32_govern_b', 'send_fatturapa', 'send_efaturagov', 'send_chorus', 'send_espap','send_fe_ap', 'andorra']
    trans.replace({w: "B2G" for w in B2G}, inplace=True)

    # Peppol
    pep = ['peppol_dh_ubl', 'peppol', 'peppol_xrechnung', 'peppol_nl_cius', 'peppol_zugferd', 'peppol_anz','peppolbis21','peppolbis30','ublinvoice_20', 'chorus_ubl']
    trans.replace({w: "Peppol" for w in pep},inplace=True)



    #Mail
    mail = ['xml_by_mail_facturae32','xml_by_mail_fatturapa','xml_by_mail_xrechnung','xml_by_mail_zugferd',
            'xml_by_mail_chorus_ubl','xml_by_mail_nlcius_ubl','xml_by_mail_svefaktura','pdf_by_mail','email', 'link_to_pdf_by_mail', 'link_to_facturae32_by_mail']
    trans.replace({w: "Mail" for w in mail}, inplace=True)

    #Altres xarxes
    altres = ['send_bsabadell','b2bconecta','facturae_322_faceb2b_detached','facturae_322_faceb2b', 'send_vocento','send_serhs','send_gmfood', 'faceb2b','pagero',
    'seres32','from_issued', 'signed_pdf', 'web_form', 'gefact', 'uploaded', 'wesupply', 'send_ariadne', 'sdi', 'sftp', 'uploaded_csv', 'inditex', 'send_edicom',
    'send_ediversa','send_ediversa_d96a','send_bonpreu','send_carrefour','send_carrefour_socomo','send_carrefour_mercancias','send_eci','edicom','send_eprior_prod',
    'paper', 'carrefour', 'carrefour-supeco','new','carrefour-champion','carrefour-hipermercados',
    'carrefour-mercancias','carrefour-plataformas','carrefour-socomo','carrefour-supersol' ]
    trans.replace({w: "Other" for w in altres},inplace=True)

    #Downloads
    down = ['download_svefaktura','download_signed_pdf','download_pdf','download_ubl','download_nlcius_ubl','download_chorus_ubl','download_zugferd','download_xrechnung','download_fatturapa','download_facturae']
    trans.replace({w: "Downloads" for w in down}, inplace=True)




    #B2Brouter
    trans.replace({'b2brouter': 'B2Brouter'}, inplace=True)



    return trans
def foo(x):
    m = pd.Series.mode(x)
    return m.values[0] if not m.empty else np.nan


def channel_agrupation(trans):

    international_sent = trans.groupby('sender_xin_value').agg({"international":"sum"})
    international_sent.columns = ['international_sent']
    international_sent.index.name = 'xin_value'

    international_rec = trans.groupby('receiver_xin_value').agg({"international":"sum"})
    international_rec.columns = ['international_rec']
    international_rec.index.name = 'xin_value'

    channel_sent = trans.groupby('sender_xin_value').agg({'channel':foo})
    channel_sent.columns = ['channel_sent']
    channel_sent.index.name = 'xin_value'

    channel_rec = trans.groupby('receiver_xin_value').agg({'channel':foo})
    channel_rec.columns = ['channel_rec']
    channel_rec.index.name = 'xin_value'


    international_sent = trans.groupby('sender_xin_value').agg({"international":"sum"})
    international_sent.columns = ['international_sent']
    international_sent.index.name = 'xin_value'

    international_rec = trans.groupby('receiver_xin_value').agg({"international":"sum"})
    international_rec.columns = ['international_rec']
    international_rec.index.name = 'xin_value'

    channel_sent = trans.groupby('sender_xin_value').agg({'channel':foo})
    channel_sent.columns = ['channel_sent']
    channel_sent.index.name = 'xin_value'

    channel_rec = trans.groupby('receiver_xin_value').agg({'channel':foo})
    channel_rec.columns = ['channel_rec']
    channel_rec.index.name = 'xin_value'

    international = pd.concat([international_sent,international_rec,channel_sent,channel_rec], axis =1)
    international.index.name = 'xin_value'

    return international



def frequency(transactions, transactions_,real_companies):
    '''
    The monthly number of transactions
    '''
    transactions_all = transactions.copy()
    transactions_received = transactions[(transactions['state']=='received') | (transactions['kind']=='reception')]
    transactions_sent = transactions[(transactions['state']=='sent') | (transactions['kind']=='issue')]
    #real_companies = list(transactions_received['receiver_xin_value']) + list(transactions_sent['sender_xin_value'])


    frequency_received = transactions_all.groupby(['receiver_xin_value']).size().reset_index(name='trans_sum_received')
    frequency_received = frequency_received.set_index('receiver_xin_value')
    frequency_received.index.name = 'xin_value'

    frequency_sent = transactions_all.groupby(['sender_xin_value']).size().reset_index(name='trans_sum_sent')
    frequency_sent = frequency_sent.set_index('sender_xin_value')
    frequency_sent.index.name = 'xin_value'


    if rfm==False:
        frequency_received_ = transactions_all.groupby(['timestamp_ym','receiver_xin_value']).size().reset_index(name='trans_med_received')
        frequency_received_ = frequency_received_.groupby(['receiver_xin_value']).agg({'trans_med_received':'median'})
        frequency_received_.index.name = 'xin_value'

        frequency_received__ = transactions_all.groupby(['timestamp_ym','receiver_xin_value']).size().reset_index(name='trans_max_received') #
        frequency_received__ = frequency_received__.groupby(['receiver_xin_value']).agg({'trans_max_received':'max'})
        frequency_received__.index.name = 'xin_value'

        frequency_sent_ = transactions_all.groupby(['timestamp_ym','sender_xin_value']).size().reset_index(name='trans_med_sent') #
        frequency_sent_ = frequency_sent_.groupby(['sender_xin_value']).agg({'trans_med_sent':'median'})
        frequency_sent_.index.name = 'xin_value'

        frequency_sent__ = transactions_all.groupby(['timestamp_ym','sender_xin_value']).size().reset_index(name='trans_max_sent')
        frequency_sent__ = frequency_sent__.groupby(['sender_xin_value']).agg({'trans_max_sent':'max'})
        frequency_sent__.index.name = 'xin_value'

        frequency = pd.concat([frequency_received_,frequency_received__,frequency_received,frequency_sent_,frequency_sent__,frequency_sent],axis=1)

    else:
        frequency = pd.concat([frequency_received,frequency_sent],axis=1)

    frequency = frequency.fillna(0)

    frequency = frequency[frequency.index.isin(real_companies)]

    return frequency

def recency(transactions, datee):
    '''
    Get the days that have passed since the last transaction (if churn, then difference between baixa and last trans)
    '''
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], utc=True)

    recency = transactions.groupby(['sender_xin_value']).agg({'timestamp':'last'})
    recency['days_from_last_transaction'] = [0]*len(recency)
    for i in range(len(recency)):
        recency['days_from_last_transaction'][recency.index[i]] = difference_dates(datee,recency['timestamp'][recency.index[i]])
    recency.index.name = 'xin_value'
    recency = recency.drop(columns='timestamp')


    recency2 = transactions.groupby(['receiver_xin_value']).agg({'timestamp':'last'})
    recency2['days_from_last_transaction_rec'] = [0]*len(recency2)
    for i in range(len(recency2)):
        recency2['days_from_last_transaction_rec'][recency2.index[i]] = difference_dates(datee,recency2['timestamp'][recency2.index[i]])
    recency2.index.name = 'xin_value'
    recency2 = recency2.drop(columns='timestamp')

    recency = pd.concat([recency,recency2], axis=1)
    return recency



def money(transactions, transactions_,actives):
    transactions['import'] = abs(transactions['import'])
    transactions_['import'] = abs(transactions_['import'])

    transactions_all = transactions.copy()

    real_companies = list(actives)

    rec_med = transactions_all.groupby(['receiver_xin_value'])['import'].median().reset_index(name='import_med_rec')
    rec_med = rec_med.set_index('receiver_xin_value')
    rec_med.index.name = 'xin_value'

    rec_med_t = transactions_.groupby(['receiver_xin_value'])['import'].median().reset_index(name='med_import_rec_t')
    rec_med_t = rec_med_t.set_index('receiver_xin_value')
    rec_med_t.index.name = 'xin_value'

    rec_sum = transactions_all.groupby(['receiver_xin_value'])['import'].sum().reset_index(name='import_sum_rec')
    rec_sum = rec_sum.set_index('receiver_xin_value')
    rec_sum.index.name = 'xin_value'

    rec_max = transactions_all.groupby(['receiver_xin_value'])['import'].max().reset_index(name='import_max_rec')
    rec_max = rec_max.set_index('receiver_xin_value')
    rec_max.index.name = 'xin_value'



    sent_med = transactions_all.groupby(['sender_xin_value'])['import'].median().reset_index(name='import_med_sent')
    sent_med = sent_med.set_index('sender_xin_value')
    sent_med.index.name = 'xin_value'

    sent_med_t = transactions_.groupby(['sender_xin_value'])['import'].median().reset_index(name='med_import_sent_t')
    sent_med_t = sent_med_t.set_index('sender_xin_value')
    sent_med_t.index.name = 'xin_value'

    sent_sum = transactions_all.groupby(['sender_xin_value'])['import'].sum().reset_index(name='import_sum_sent')
    sent_sum = sent_sum.set_index('sender_xin_value')
    sent_sum.index.name = 'xin_value'

    sent_max = transactions_all.groupby(['sender_xin_value'])['import'].max().reset_index(name='import_max_sent')
    sent_max = sent_max.set_index('sender_xin_value')

    sent_max.index.name = 'xin_value'
    money = pd.concat([rec_med,rec_max,rec_sum,sent_med,sent_max,sent_sum],axis=1)

    money = money.fillna(0)
    money = money[money.index.isin(real_companies)]

    money_ = sent_med_t.join(rec_med_t, how='outer')
    money_ = money_[money_.index.isin(real_companies)]

    money_=money_.join(money, how='outer')

    return money_

#%%
# REVENUE
# only companies created after 201907
revenue = revenue_info()
if rfm ==False:

    revenue = revenue[revenue['timestamp_ym']>='201907']
revenue_ = revenue[((revenue['action']=='alta') & (revenue['plan']=='basic'))]
revenue_ = revenue_.groupby('company').agg({'action':'last'})
ids = revenue_.index
revenue = revenue[revenue['company'].isin(ids)]

if rfm ==False:

    revenue = revenue[revenue['timestamp_ym']<'202203'] #
revenue = revenue.sort_values(['@timestamp', 'date_last_updated'], ascending=[True,True])
#%%
revenue = replace_plans(revenue)
revenue['plan'] = revenue['plan'].fillna(-1)
revenue['plan'] = revenue['plan'].astype('int')
#%%
#TRANSACTIONS
trans = pd.read_csv('transs.csv')
trans = trans.drop(columns='Unnamed: 0')
if rfm ==False:
    trans = trans[trans['timestamp_ym']<202203]  ########################
#%%
timestamp_ym = pd.to_datetime(trans['timestamp']).dt.strftime('%Y%m')
trans['timestamp_ym'] = timestamp_ym
#%%
#trans = transactions_info()

df_backup = trans.copy()
trans = replace_plans(trans)
trans['company_plan'] = trans['company_plan'].fillna(-1)
trans = trans[(trans['company_plan']!='enterprise') & (trans['company_plan']!='aplicateca2') & (trans['company_plan']!='aplicateca5')]
trans['company_plan'] = trans['company_plan'].astype('int')
trans = trans[trans['company_id'].isin(ids)]


trans = trans[trans['sender_xin_value']!='#personal num']
trans = trans[trans['receiver_xin_value']!='#personal num']
trans = trans.sort_values(['timestamp'], ascending=[True])
#%%
trans['channel_original'] = trans['channel']
#%%
trans = channel_replacement(trans)
#%%
#COMPANY:
company = company_info()
if rfm ==False:
    company = company[company['date_last_updated']<'2022-03-01 00:00:00+00:00']
company = replace_plans(company)
company = company[company['company_id'].isin(ids)]

companies_XINS = company[['company_id','xin_value']]
companies_XINS = companies_XINS.set_index('xin_value')
companies_XINS = companies_XINS.dropna()
#%%
#DATASETS CREATED:
active_premiums = active_prem(revenue)
companies = all_users(revenue,company)
companies = companies.join(active_premiums)

#%%AQUIIIIIIIII
#relation xin-id
xinss = trans[~trans['company_id'].isin(companies_XINS['company_id'])]
#%%
xinss['channel_original'] = xinss['channel_original'].fillna('')
peppol = xinss[xinss['channel_original'].str.contains('peppol')]
sents = xinss[(xinss['channel_original'].str.contains('peppol') ==False) &
            ((xinss['state'] == 'sent') |
            (xinss['kind']=='issue') & (xinss['state']!='paid') |
            (xinss['kind']=='issue') & (xinss['state']!='accepted') )]
recieves = xinss[(xinss['channel_original'].str.contains('peppol') ==False) &
                (((xinss['state']=='received') |
                (xinss['kind']=='reception')|
                ((xinss['kind']=='issue') & (xinss['state']=='paid') |
                (xinss['kind']=='issue') & (xinss['state']=='accepted'))
                ))]

companies_xins1 = sents.groupby('sender_xin_value').agg({'company_id':set})
companies_xins1.index.name = 'xin_value'
companies_xins2 = recieves.groupby('receiver_xin_value').agg({'company_id':set})
companies_xins2.index.name = 'xin_value'
#%%

companies_xins1 = sents.groupby('sender_xin_value').agg({'company_id':'max'})
companies_xins1.index.name = 'xin_value'
companies_xins2 = recieves.groupby('receiver_xin_value').agg({'company_id':'max'})
companies_xins2.index.name = 'xin_value'

companies_xins = companies_xins1.append(companies_xins2)

companies_xins = companies_xins.groupby('xin_value').agg({'company_id' : 'max'})
companies_xins = companies_xins.append(pd.DataFrame(105288, index=['#personal xin'], columns = ['company_id'])).append(pd.DataFrame(151379, index=['#personal xin'], columns = ['company_id']))
#%%
companies_xins.index.name = 'xin_value'
companies_XINS = companies_xins.append(companies_XINS)
#%%
companies_XINS = companies_XINS.reset_index()
companies_XINS = companies_XINS.set_index('company_id')
companies_XINS = companies_XINS.fillna('IN')
companies_XINS['xin_value'] = companies_XINS['xin_value'].apply(lambda x: processNan(x) if x is 'IN' else x)

#%%
companies_XINS.to_csv('companies_XINS.csv')
#%%
churn = churn_dataset(revenue)
#%%
churn_b = churn_basic(revenue)
#%%


#%%
def TFG():
    trans['timestamp_ym'] = trans['timestamp_ym'].astype(int)
    company['date_last_updated_ym'] = company['date_last_updated_ym'].astype(int)
    revenue['timestamp_ym'] = revenue['timestamp_ym'].astype(int)

    timestamp_ym = list((trans.groupby('timestamp_ym').agg({'id':'last'})).index)
    timestamp_ym = map(int, timestamp_ym)

    all=[]
    last = 201906
    for i in timestamp_ym:
        print(i)
        revenue_ = revenue[(revenue['timestamp_ym'] <= i)]
        activess = actives(revenue_)
        activess = activess.join(companies_XINS, how = 'left')
        activess = list(activess['xin_value'])

        if rfm == True:
            trans_ = trans[(trans['timestamp_ym'] <= i) & (trans['timestamp_ym'] > last)]
        else:
            trans_ = trans[(trans['timestamp_ym'] <= i) & (trans['timestamp_ym'] > i-3)]
        last_trans_ = trans[(trans['timestamp_ym'] <= i)]

        plan_actual = last_plan(revenue_)
        dataset = plan_actual.copy()
        dataset.index.name = 'company_id'

        dataset = dataset.join(companies_XINS)
        dataset = dataset.reset_index()
        dataset = dataset.set_index('xin_value')

        products = transactions_byplan(trans_)

        plan_last_trans = last_transaction_plan(last_trans_)

        number_receivers_senders = num_rec_sent(last_trans_) ###

        succeeded_trans = trans_success(trans_) ###

        #get current last date
        datee = str(i)[4:7] + '-' + '01' + '-' + str(i)[0:4]
        datee = pd.to_datetime(datee) + pd.DateOffset(months=1)
        datee = datee.strftime('%Y-%m-%d %H:%M:%S')
        datee = pd.to_datetime(datee, utc=True)

        recencies = recency(last_trans_, datee)

        frequencies = frequency(trans_,last_trans_,activess)

        moneys = money(trans_,last_trans_,activess)

        channels = channel_agrupation(trans_)


        dataset_ = dataset.join(number_receivers_senders, how='left').join(products, how='left').join(plan_last_trans, how='left').join(succeeded_trans, how='left').join(recencies, how='left').join(frequencies, how='left').join(moneys, how='left').join(channels, how='left')

        dataset_ = dataset_.drop(dataset_[dataset_['company_id'].isna()==True].index)

        dataset = dataset_.set_index('company_id')

        dataset = time_since_alta(dataset,datee, companies,churn)



        #fillna
        dataset[[0,1,2]] = dataset[[0,1,2]].fillna(0)
        dataset[['last_trans_plan', 'last_trans_plan_rec']] = dataset[['last_trans_plan', 'last_trans_plan_rec']].fillna(-1)
        #dataset['actual_plan'] = dataset['actual_plan'].fillna(-1)
        dataset[['num_receivers','num_senders']] = dataset[['num_receivers','num_senders']].fillna(0)
        dataset[['success_true','success_false']] = dataset[['success_true','success_false']].fillna(0)
        #dataset[list(channels.columns)] = dataset[list(channels.columns)].fillna(0)
        dataset[list(frequencies.columns)] = dataset[list(frequencies.columns)].fillna(0)
        #dataset['days_from_last_transaction'] = dataset['days_from_last_transaction'].fillna(-1)
        dataset[list(moneys.columns)] = dataset[list(moneys.columns)].fillna(0)
        dataset[list(channels.columns)] = dataset[list(channels.columns)].fillna(0)

        dataset = dataset[(dataset.index).isin(ids)]

        all.append(dataset)

        last = i

    timestamp_ym = list((trans.groupby('timestamp_ym').agg({'id':'last'})).index)

    for i in range(len(all)):
        if i == 0:
            all[i]['timestamp_ym'] = timestamp_ym[i]
            joined = all[i]
        else:
            all[i]['timestamp_ym'] = timestamp_ym[i]
            joined = pd.concat([joined,all[i]],axis=0)

    return joined

#%%
def acumulated_columns(transactions_behaviour):

    #TRANSACTIONS

    transactions_behaviour['med_trans_sent'] = list(transactions_behaviour.groupby('company')['trans_sum_sent'].apply(lambda x: x.expanding().median()))
    transactions_behaviour['med_trans_received'] = list(transactions_behaviour.groupby('company')['trans_sum_received'].apply(lambda x: x.expanding().median()))

    if rfm==False:
        transactions_behaviour['med_trans_sent_m'] = list(transactions_behaviour.groupby('company')['trans_med_sent'].apply(lambda x: x.expanding().median()))
        transactions_behaviour['med_trans_received_m'] = list(transactions_behaviour.groupby('company')['trans_med_received'].apply(lambda x: x.expanding().median()))
        transactions_behaviour['max_trans_sent'] = transactions_behaviour.groupby(['company'])['trans_max_sent'].cummax()
        transactions_behaviour['max_trans_received'] = transactions_behaviour.groupby(['company'])['trans_max_received'].cummax()


    if rfm == True:
        transactions_behaviour['sum_trans_sent'] = transactions_behaviour.groupby(['company'])['trans_sum_sent'].cumsum()
        transactions_behaviour['sum_trans_received'] = transactions_behaviour.groupby(['company'])['trans_sum_received'].cumsum()
        transactions_behaviour['acum_0'] = transactions_behaviour.groupby(['company'])[0].cumsum()
        transactions_behaviour['acum_1'] = transactions_behaviour.groupby(['company'])[1].cumsum()
        transactions_behaviour['acum_2'] = transactions_behaviour.groupby(['company'])[2].cumsum()
        transactions_behaviour['acum_success_true'] = transactions_behaviour.groupby(['company'])['success_true'].cumsum()
        transactions_behaviour['acum_success_false'] = transactions_behaviour.groupby(['company'])['success_false'].cumsum()
        transactions_behaviour['max_trans_sent'] = transactions_behaviour.groupby(['company'])['trans_sum_sent'].cummax()
        transactions_behaviour['max_trans_received'] = transactions_behaviour.groupby(['company'])['trans_sum_received'].cummax()


    transactions_behaviour['med_import_sent'] = list(transactions_behaviour.groupby('company')['import_sum_sent'].apply(lambda x: x.expanding().median()))
    transactions_behaviour['med_import_rec'] = list(transactions_behaviour.groupby('company')['import_sum_rec'].apply(lambda x: x.expanding().median()))

    transactions_behaviour['max_import_sent'] = transactions_behaviour.groupby(['company'])['import_max_sent'].cummax()
    transactions_behaviour['max_import_received'] = transactions_behaviour.groupby(['company'])['import_max_rec'].cummax()

    if rfm == True:
        transactions_behaviour['sum_import_sent'] = transactions_behaviour.groupby(['company'])['import_sum_sent'].cumsum()
        transactions_behaviour['sum_import_rec'] = transactions_behaviour.groupby(['company'])['import_sum_rec'].cumsum()


    transactions_behaviour = pd.get_dummies(transactions_behaviour, columns=['channel_sent','channel_rec'])
    channels_names = ['b2g','mail','downloads','other','peppol', 'b2brouter']
    channels_names_sent = ['channel_sent_B2G','channel_sent_Mail', 'channel_sent_Downloads','channel_sent_Other','channel_sent_Peppol', 'channel_sent_B2Brouter']
    channels_names_rec = ['channel_rec_B2G','channel_rec_Mail', 'channel_rec_Downloads','channel_rec_Other','channel_rec_Peppol', 'channel_rec_B2Brouter']
    for i in range(len(channels_names)):
        transactions_behaviour[channels_names[i]] = transactions_behaviour[channels_names_sent[i]] + transactions_behaviour[channels_names_rec[i]]

    transactions_behaviour = transactions_behaviour.drop(columns=channels_names_sent)
    transactions_behaviour = transactions_behaviour.drop(columns=channels_names_rec)
    transactions_behaviour = transactions_behaviour.drop(columns=['channel_sent_0','channel_rec_0'])

    channels_acum =[]
    for i in channels_names:
        channels_acum.append( i +'_acum' )
    transactions_behaviour[channels_acum] = transactions_behaviour.groupby(['company'])[channels_names].cumsum()


    return transactions_behaviour


#%%
transactions_behaviour = TFG()
#%%

transactions_behaviour.index.name ='company'
transactions_behaviour = transactions_behaviour.sort_values(['company', 'timestamp_ym'], ascending=[True,True]) #per fer la median rolling
#%%
transactions_behaviour_ = acumulated_columns(transactions_behaviour.copy()) #transactions_behaviour_
#%%
#%%
if rfm==False:
    transactions_behaviour_.to_csv('transactions_behaviour_t.csv')
    companies.to_csv('companies.csv')


if rfm==True:
    transactions_behaviour_.to_csv('transactions_behaviour.csv')
    companies.to_csv('companies.csv')
    churn.to_csv('churn.csv')
    churn_b.to_csv('churn_basic.csv')

# %%
