# data manipulation
import pandas as pd
import numpy as np
import scipy
import pyodbc

# visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# machine learning
from sklearn import preprocessing, utils, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFECV, chi2, mutual_info_classif
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# umap 
import umap

# others
import os
import copy
from tqdm import tqdm
import datetime


# matlpotlib properties
plt.rcParams["figure.figsize"] = 6,6
plt.rcParams["figure.dpi"] = 300
plt.rcParams['figure.max_open_warning'] = False
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# seaborn properties
sns.set(style="whitegrid")


##########################################################################################################################
# get the dataset B00 TrgPos
# extracts all datasets from database with selected prefix
##########################################################################################################################

def extractDataset(prefix, excludeSet):
    conn = pyodbc.connect("DRIVER={SQL Server Native Client 11.0};SERVER=VHACDWRB03.vha.med.va.gov;DATABASE=ORD_Singh_201911038D;Trusted_Connection=yes")

    info_df = pd.read_sql(sql="select * from information_schema.tables where table_name like '%{}%'".format(prefix), con=conn)
    display(info_df)
    # read all the tables into pandas tables
    tables = {}
    for tname in info_df.TABLE_NAME:
        if tname.split('_')[-1] in excludeSet:
            continue
        query_str = "select * from  " + str("Dflt.")+ str(tname)
        table_df = pd.read_sql(sql=query_str,con=conn)
        tables[tname.split('_')[-1]] = table_df
    print(tables.keys())
    return tables


#########################################################################################################################
# make clean cohort dataframe
#########################################################################################################################

# for every (PtSSN,EDStartDateTime) pair
#   - ED duration (hours)
#   - time to first hospitalization after EDEndDateTime(days)
#   - length of hospital stays (days) [sum of all stays associated with index ED visit]
#   - number of hospitalizations

def make_cohort_df(cohort):
    # standardize PtSSN
    cohort.rename(columns={'patientSSN':'PtSSN','B':'DischargeDateTime'},inplace=True)
    cohort.PtSSN = cohort.PtSSN.astype('int64')
    
    cohort['hosp_stay'] = (cohort.DischargeDateTime - cohort.AdmitDateTime)/np.timedelta64(1,'D')
    cohort['ed_duration'] = (cohort.EDEndDateTime - cohort.EDStartDateTime)/np.timedelta64(1,'h')
    
    # create num ED visits for every PtSSN, EDStartDateTime pair
    tmp = cohort1.groupby(['PtSSN'])[['EDStartDateTime']].agg('count')
    tmp.rename(columns={'EDStartDateTime':'num_ED_visits'},inplace=True)
    tmp = tmp.reset_index()
    cohort2 = pd.merge(cohort1,tmp,on=['PtSSN'])
    
    # create number of hospitalizations
    tmp = cohort2.groupby(['PtSSN','EDStartDateTime'])[['AdmitDateTime']].agg('count')
    tmp.rename(columns={'AdmitDateTime':'num_hosp'},inplace=True)
    tmp = tmp.reset_index()
    cohort3 = pd.merge(cohort2,tmp,on=['PtSSN','EDStartDateTime'])
    
    # select records with num_hosp == 2 and check that they are just a few minutes apart
    # treat them as a new admissions with Admit date from 1st record and Discharge date from 2nd record
    # drop the two original records
    
    hosp2_cohort = cohort3[cohort3.num_hosp==2]
    
    
    # create ed_first_inp_delta for every PtSSN, EDStartDateTime pair
    tmp = cohort.groupby(['PtSSN','EDStartDateTime'])[['AdmitDateTime']].agg('min')
    tmp.rename(columns={'AdmitDateTime':'FirstAdmission'},inplace=True)
    tmp = tmp.reset_index()
    cohort1 = pd.merge(cohort,tmp,on=['PtSSN','EDStartDateTime'])
    cohort1['ed_first_inp_delta'] = (cohort1.FirstAdmission - cohort1.EDEndDateTime)/(np.timedelta64(1,'D'))
    
    
    
    # create hosp_stay min, max, mean for every PtSSN,EDStartDateTime pair
    tmp = cohort3.groupby(['PtSSN','EDStartDateTime'])[['hosp_stay']].agg(['min','max','mean'])
    tmp.columns = [x[0] + '_' + x[1] for x in tmp.columns.to_flat_index()]
    cohort4 = pd.merge(cohort3,tmp,on=['PtSSN','EDStartDateTime'])
    
    # create ed_duration min, max,mean for every PtSSN
    tmp = cohort4.groupby(['PtSSN'])[['ed_duration']].agg(['min','max','mean'])
    tmp.columns = [x[0] + '_' + x[1] for x in tmp.columns.to_flat_index()]
    cohort5 = pd.merge(cohort4,tmp,on=['PtSSN'])
    
    return cohort5

#########################################################################################################################
# Separate dizzy and abdpain cohorts
#########################################################################################################################

def separate_cohorts(cohorts):
    abdpain_cohort = cohorts[cohorts.TriggerType=='AbdominalPain'].copy()
    dizzy_cohort = cohorts[cohorts.TriggerType=='Dizziness'].copy()
    print('Dizzy cohort = ', dizzy_cohort.shape,' Abdpain cohort = ',abdpain_cohort.shape)

    dizzy_cohort_df = make_cohort_df(dizzy_cohort)
    abdpain_cohort_df = make_cohort_df(abdpain_cohort)
    print('Dizzy cohort df = ', dizzy_cohort_df.shape,' Abdpain cohort df = ',abdpain_cohort_df.shape)

  
    return dizzy_cohort_df, abdpain_cohort_df

#########################################################################################################################
# Get labeled data
#########################################################################################################################

dizzy_fname = "../ML Final data/ML_4A_Dizziness/csv format/Ashish_HighRiskDizziness_ValidationForm_Sep2021(n=100)_v2_VV1.csv"
abdpain_fname = "../ML Final data/ML_4B_Abdominal Pain/csv format/Ashish_HighRiskAbdominalPain(n=120)_09222022_VV_v1.csv"


def retrieveLabels_dizzy(fname):
    """"
    retrieves labels for the ML4 Dizziness trigger
    
    Returns
    --------
    dataframe
         of reviewed entries
    dict
         label dictionary mapping ssn to label
         
    """
    dizzy_df = pd.read_csv(open(fname, 'r', errors="ignore"))
    dizzy_df.dropna(subset=["DxErrorERCoded", "PtSSN"], inplace=True)
    
    label_map = dict()
    
    for index, row in dizzy_df.iterrows():
        if row["PtSSN"] in label_map:
            print("duplicate label???")
        
        
        label_map[row["PtSSN"]] = row['DxErrorERCoded']
            
    labels_df = pd.DataFrame.from_dict(label_map,orient='index').reset_index()
    labels_df.columns=['PtSSN','label']
    
    
    return dizzy_df, labels_df

def retrieveLabels_abdpain(fname):
    """"
    retrieves labels for the ML4 AbdominalPain trigger
    
    Returns
    --------
    dataframe
         of reviewed entries
    dict
         label dictionary mapping ssn to label
         
    """
    abpain_df = pd.read_csv(open(fname, 'r', errors="ignore"))
   
    abpain_df.dropna(subset=["DxError", "PtSSN"], inplace=True)
   
    abpain_df["PtSSN"] = abpain_df["PtSSN"].astype(int)
    label_map = dict()
    
    for index, row in abpain_df.iterrows():
        if row["PtSSN"] in label_map:
            print("duplicate label???")
        if row["DxError"] in [1,2,3]:
            label_map[row["PtSSN"]] = "NoMOD"
        if row["DxError"] in [5,6]:
            label_map[row["PtSSN"]] = "MOD"
            
    labels_df = pd.DataFrame.from_dict(label_map,orient='index').reset_index()
    labels_df.columns=['PtSSN','label']
    
    return abpain_df, labels_df



#########################################################################################################################
# do a ttest between MOD/NoMOD on a specified set of fields
#########################################################################################################################
# make sure to dropna before running the ttest
# currently only tests continuous fields
def ttest_fields(df,fields,ftypes,show=False):
    sig_fields = []
    #tmp = df.dropna()
    tmp1 = df[df.label.isin(['MOD','NoMOD'])]
    for i in range(len(fields)):
        field = fields[i]
        if ftypes[i] == 'c':
            tstat,pval = scipy.stats.ttest_ind(tmp1[tmp1.label=='MOD'][field],tmp1[tmp1.label=='NoMOD'][field],equal_var=False )
        if ftypes[i] == 'd':
            contingency = pd.crosstab(tmp1[field],tmp1['label'])
            tstat, pval, _, _ = scipy.stats.chi2_contingency(contingency)
            
        if show:
            print(field,tstat,pval)
        if pval <= 0.1:
            sig_fields.append(field)
    return sig_fields
        
def violin_plot_field(df,field,title):
    fldist_fig, fldist_axes = plt.subplots()
    sns.violinplot(data=df, x=field, y='label', scale='count', inner='stick', ax=fldist_axes)
    # Set text
    fldist_axes.set_title(title + ' ' + field + ' Distribution')
    fldist_axes.set_ylabel('')
    fldist_axes.set_xlabel('')
    fldist_fig.set_figheight(6)
    fldist_fig.set_figwidth(6)
        

#########################################################################################################################
# code the demographics
#########################################################################################################################
def separate_cohorts_demo(demog,dizzy_pts,abdpain_pts):
    
    # drop demog records that have null DOB values
    demog = demog.dropna(subset=['DOB']).copy()

    # standardize PtSSN field
    demog.rename(columns={'patientSSN':'PtSSN'},inplace=True)
    demog.PtSSN = demog.PtSSN.astype('int64')

    dizzy_demo = demog[demog.PtSSN.isin(dizzy_pts)].copy()
    print('dizzy demographics shape = ', dizzy_demo.shape)

    abdpain_demo = demog[demog.PtSSN.isin(abdpain_pts)].copy()
    print('abdpain demographics shape = ', abdpain_demo.shape)
    
    return dizzy_demo,abdpain_demo

def clean_race_values(x):
    if (x=='DECLINED TO ANSWER') | (x=='UNKNOWN BY PATIENT'):
        return 'UNKNOWN'
    if (x=='AMERICAN INDIAN OR ALASKA NATIVE') | (x=='NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'):
        return 'ALASKA/PACIFIC'
    return x

def make_demo_df(demo):
    
    # convert date fields to pandas format
    demo.DOB = pd.to_datetime(demo.DOB)
    demo.IndexDateTime = pd.to_datetime(demo.IndexDateTime)
    
    # add age at index visit
    demo['age_at_index_visit'] = np.round((demo['IndexDateTime'] - demo['DOB'])/np.timedelta64(1,'Y'),2)
    
    # clean up race field
    demo['new_race'] = demo.Race.apply(lambda x: clean_race_values(x))
    
    # pull out relevant demographic fields
    demo_df = demo[['PtSSN','age_at_index_visit','Gender','Ethnicity','new_race']].copy()
    
    return demo_df

# Make demographic data frame
# - age_at_index_visits (yrs)
# - ethncitiy: 1 = latino/hispanic and 0 = not
# - gender: 1 = male and 0 = female
# - race: alaska/pacific, asian, black, unknown, white


def code_demodf(demo_df):
    # code Gender, Ethnicity, new_race 
    tmp1 = pd.get_dummies(demo_df[['new_race']])
    tmp2 = pd.DataFrame(demo_df['Gender'].apply(lambda x: 1 if x=='M' else 0))
    tmp3 = pd.DataFrame(demo_df['Ethnicity'].apply(lambda x: 1 if x=='HISPANIC OR LATINO' else 0))
    demo_coded = pd.concat([demo_df[['PtSSN','age_at_index_visit']],tmp1,tmp2,tmp3],axis=1)
    return demo_coded

#########################################################################################################################
# Separate and code demographic data
#########################################################################################################################

def separate_demog(demog,dizzy_cohort_df,abdpain_cohort_df):
    dizzy_demo,abdpain_demo = separate_cohorts_demo(demog,dizzy_cohort_df.PtSSN.unique(),abdpain_cohort_df.PtSSN.unique())

    dizzy_demo_df = make_demo_df(dizzy_demo)
    abdpain_demo_df = make_demo_df(abdpain_demo)
    #print(dizzy_demo_df.shape,abdpain_demo_df.shape)
    
    # code the demographic dataframe
    dizzy_demo_coded = code_demodf(dizzy_demo_df)
    abdpain_demo_coded = code_demodf(abdpain_demo_df)
    return dizzy_demo_coded, abdpain_demo_coded

#########################################################################################################################
# UMAP plots
#########################################################################################################################
def umap_plot_nolabel(X,rseed,nbrs):
    # standard scale X
    scaler = StandardScaler()
    
    #sX = scaler.fit_transform(X.astype(float))
    sX = scaler.fit_transform(X.values)
    
    reducer = umap.UMAP(metric='correlation',min_dist=0.0,n_neighbors=nbrs,random_state=rseed)
    u = reducer.fit_transform(X)
    plt.subplots(figsize=(8,8))
    plt.scatter(u[:,0],u[:,1],alpha=0.5,s=40)
    return u



def umap_plot_label(X,y,rseed,nbrs,label_list,legend_loc):
    nclasses = len(label_list)
    # standard scale X
    scaler = StandardScaler()
    #sX = scaler.fit_transform(X.astype(float))
    sX = scaler.fit_transform(X.values)
    
    reducer = umap.UMAP(metric='correlation',min_dist=0.0,n_neighbors=nbrs,random_state=rseed)
    u = reducer.fit_transform(X)
    cc = plt.cm.rainbow(np.linspace(0,1,nclasses))
    colors = {i:cc[i] for i in range(nclasses)} 
    plt.subplots(figsize=(8,8))
    for i in range(nclasses):
        plt.scatter(u[y==label_list[i],0],u[y==label_list[i],1],alpha=0.5,s=40,label=label_list[i],color=colors[i])

    plt.legend(loc=legend_loc)
    return u

# characterize the clusters in the labeled data
# cluster the embedding and find the cluster composition

def cluster_umap(u,N,df,loc):
    clust = KMeans(n_clusters = N)
    clust.fit(u)

    plt.subplots(figsize=(8,8))
    for i in range(N):
        plt.scatter(u[clust.labels_==i,0],u[clust.labels_==i,1])
    plt.legend([i+1 for i in range(N)],loc=loc)
    
    for i in range(N):
        print('Cluster ',i+1)
        print(df.iloc[np.where(clust.labels_==i)[0],:].label.value_counts())
    return clust

def cluster_umap_nolabel(u,N,df,loc):
    clust = KMeans(n_clusters = N)
    clust.fit(u)

    plt.subplots(figsize=(8,8))
    for i in range(N):
        plt.scatter(u[clust.labels_==i,0],u[clust.labels_==i,1])
    plt.legend([i+1 for i in range(N)],loc=loc)
    
  
    return clust

# produce median statistics of cluster 
def analyze_clusters(clust,df,feats):
    N = clust.n_clusters
    clusters = pd.DataFrame(data=clust.labels_,columns=["cluster"])
    
    # need to scale the df for tapestry plot
    scaler = StandardScaler()
    sX = scaler.fit_transform(df[feats])
    scaled_df = pd.DataFrame(data=sX,columns=feats)
    
    median_clusters =np.zeros((clust.n_clusters,len(feats)))
    for i in range(N):
        print('median statistics of cluster ', i+1)
        print(df.iloc[np.where(clust.labels_==i)[0],:].label.value_counts())
        median_clusters[i,:] = scaled_df[clust.labels_==i].median()
        display(df[clust.labels_==i].median())
     
    
    median_clusters_df = pd.DataFrame(median_clusters,columns=feats)   
    plt.figure(figsize=(15,3))
    plt.title('Medians for each cluster')
    plt.imshow(median_clusters_df,cmap='jet')
    plt.xticks(ticks=np.arange(median_clusters_df.shape[1]),labels=median_clusters_df.columns,rotation=90)
    plt.colorbar(shrink=0.5)
    plt.yticks(ticks=np.arange(median_clusters_df.shape[0]),labels=["Cluster "+ str(i+1) for i in range(median_clusters_df.shape[0])])
          
   
        
def analyze_clusters_nolabel(clust,df,feats):
    N = max(clust.labels_)
    clusters = pd.DataFrame(data=clust.labels_,columns=["cluster"])
    
    # need to scale the df for tapestry plot
    scaler = StandardScaler()
    sX = scaler.fit_transform(df[feats])
    scaled_df = pd.DataFrame(data=sX,columns=df.columns)
    
    median_clusters =np.zeros((clust.n_clusters,df.shape[1]))
    for i in range(N+1):
        print('median statistics of cluster ', i)
        display(df[clust.labels_==i].describe()[feats].T['50%'])
        median_clusters[i,:] = scaled_df[clust.labels_==i].median()
    median_clusters_df = pd.DataFrame(median_clusters,columns=df.columns)
    
    plt.figure(figsize=(15,3))
    plt.title('Medians for each cluster')
    plt.imshow(median_clusters_df,cmap='jet')
    plt.xticks(ticks=np.arange(median_clusters_df.shape[1]),labels=median_clusters_df.columns,rotation=90)
    plt.colorbar(shrink=0.5)
    plt.yticks(ticks=np.arange(median_clusters_df.shape[0]),labels=["Cluster "+ str(i) for i in range(median_clusters_df.shape[0])])
        
#########################################################################################################################
# Vitals data
#########################################################################################################################

def add_EDvital_column(vitals,vital_col,colname):
    tmp = vitals.groupby(['PtSSN','EDStartDateTime'])[[vital_col]].agg(['count','max','min','first'])
    tmp = tmp.reset_index()
    tmp.columns = [colname + '_'+x[1] if x[1]!= '' else x[0] for x in tmp.columns.to_flat_index()]
    
    return tmp

def make_EDvitals_df(EDvitals_df,cohort_df):
    systolic = add_EDvital_column(EDvitals_df,'Systolic','Systolic')
    # fill out missing values
    systolic = pd.merge(cohort_df[['PtSSN','EDStartDateTime']],systolic,how='left')
    systolic['Systolic_count'].fillna(0,inplace=True)
    
    diastolic = add_EDvital_column(EDvitals_df,'Diastolic','Diastolic')
    diastolic = pd.merge(cohort_df[['PtSSN','EDStartDateTime']],diastolic,how='left')
    diastolic['Diastolic_count'].fillna(0,inplace=True)
    
    vdf = pd.merge(systolic,diastolic,on=['PtSSN','EDStartDateTime'])

    vital_types = ['PULSE','RESPIRATION','PAIN','TEMPERATURE']
    for vital_type in vital_types:
        subset_EDvitals = EDvitals_df[EDvitals_df.VitalType==vital_type]
        tmp_df = add_EDvital_column(subset_EDvitals,'VitalResultNumeric',vital_type)
        tmp_df = pd.merge(cohort_df[['PtSSN','EDStartDateTime']],tmp_df,how='left')
        tmp_df[vital_type+'_count'].fillna(0,inplace=True)
        vdf = pd.merge(vdf,tmp_df,on=['PtSSN','EDStartDateTime'])
        vdf.drop_duplicates(inplace=True)
    return vdf


# separate the two cohorts and extract only ED vitals
def separate_cohorts_EDvitals(vitals,dizzy_cohort_df,abdpain_cohort_df):
    vitals.rename(columns={'PatientSSN':'PtSSN'},inplace=True)
    vitals.PtSSN = vitals.PtSSN.astype('int64')
    
    dizzy_vitals = pd.merge(vitals,dizzy_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    dizzy_ED_vitals = dizzy_vitals[(dizzy_vitals.VitalSignTakenDateTime >= dizzy_vitals.EDStartDateTime) & 
                       (dizzy_vitals.VitalSignTakenDateTime <= dizzy_vitals.EDEndDateTime)].copy()
    abdpain_vitals = pd.merge(vitals,abdpain_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    abdpain_ED_vitals = abdpain_vitals[(abdpain_vitals.VitalSignTakenDateTime >= abdpain_vitals.EDStartDateTime) & 
                       (abdpain_vitals.VitalSignTakenDateTime <= abdpain_vitals.EDEndDateTime)].copy()
    print(dizzy_ED_vitals.shape,abdpain_ED_vitals.shape)
    
    # convert to a dataframe
    dizzy_EDvitals_df = make_EDvitals_df(dizzy_ED_vitals,dizzy_cohort_df)
    abdpain_EDvitals_df = make_EDvitals_df(abdpain_ED_vitals,abdpain_cohort_df)

    return dizzy_EDvitals_df,abdpain_EDvitals_df

def add_hosp_vital_column(vitals,vital_col,colname):
    tmp = vitals.groupby(['PtSSN','AdmitDateTime'])[[vital_col]].agg(['count','max','min','first'])
    tmp = tmp.reset_index()
    tmp.columns = ['HOSP_'+ colname + '_'+x[1] if x[1]!= '' else x[0] for x in tmp.columns.to_flat_index()]
    #print(tmp.columns)
    return tmp

def make_hosp_vitals_df(hosp_vitals_df,cohort_df):
    systolic = add_hosp_vital_column(hosp_vitals_df,'Systolic','Systolic')
    systolic = pd.merge(cohort_df[['PtSSN','AdmitDateTime']],systolic,how='left')
    systolic['HOSP_Systolic_count'].fillna(0,inplace=True)
    diastolic = add_hosp_vital_column(hosp_vitals_df,'Diastolic','Diastolic')
    diastolic = pd.merge(cohort_df[['PtSSN','AdmitDateTime']],diastolic,how='left')
    diastolic['HOSP_Diastolic_count'].fillna(0,inplace=True)
    
    vdf = pd.merge(systolic,diastolic,on=['PtSSN','AdmitDateTime'])
    

    vital_types = ['PULSE','RESPIRATION','PAIN','TEMPERATURE']
    for vital_type in vital_types:
        subset_hosp_vitals = hosp_vitals_df[hosp_vitals_df.VitalType==vital_type]
        tmp_df = add_hosp_vital_column(subset_hosp_vitals,'VitalResultNumeric',vital_type)
        tmp_df = pd.merge(cohort_df[['PtSSN','AdmitDateTime']],tmp_df,how='left')
        tmp_df['HOSP_'+vital_type + '_count'].fillna(0,inplace=True)
        vdf = pd.merge(vdf,tmp_df,on=['PtSSN','AdmitDateTime'])
    return vdf.drop_duplicates()


def separate_cohorts_hosp_vitals(vitals,dizzy_cohort_df,abdpain_cohort_df):
    vitals.rename(columns={'PatientSSN':'PtSSN'},inplace=True)
    vitals.PtSSN = vitals.PtSSN.astype('int64')
    
    dizzy_vitals = pd.merge(vitals,dizzy_cohort_df[['PtSSN','AdmitDateTime','DischargeDateTime']],on='PtSSN')
    dizzy_hosp_vitals = dizzy_vitals[(dizzy_vitals.VitalSignTakenDateTime >= dizzy_vitals.AdmitDateTime) & 
                       (dizzy_vitals.VitalSignTakenDateTime <= dizzy_vitals.DischargeDateTime)].copy()
    abdpain_vitals = pd.merge(vitals,abdpain_cohort_df[['PtSSN','AdmitDateTime','DischargeDateTime']],on='PtSSN')
    abdpain_hosp_vitals = abdpain_vitals[(abdpain_vitals.VitalSignTakenDateTime >= abdpain_vitals.AdmitDateTime) & 
                       (abdpain_vitals.VitalSignTakenDateTime <= abdpain_vitals.DischargeDateTime)].copy()
    print(dizzy_hosp_vitals.shape,abdpain_hosp_vitals.shape)
    
    # convert to dataframe
    dizzy_hosp_vitals_df = make_hosp_vitals_df(dizzy_hosp_vitals,dizzy_cohort_df)
    abdpain_hosp_vitals_df = make_hosp_vitals_df(abdpain_hosp_vitals,abdpain_cohort_df)
    
    return dizzy_hosp_vitals_df,abdpain_hosp_vitals_df




#########################################################################################################################
# Consults data
#########################################################################################################################
def separate_cohorts_consults(consults,dizzy_cohort_df,abdpain_cohort_df):
    consults.rename(columns={'PatientSSN':'PtSSN'},inplace=True)
    consults.PtSSN = consults.PtSSN.astype('int64')
    
    dizzy_consults = pd.merge(consults,dizzy_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    dizzy_consults = dizzy_consults[(dizzy_consults.requestDateTime >= dizzy_consults.EDStartDateTime) &                                 
                       (dizzy_consults.requestDateTime <= dizzy_consults.EDEndDateTime)].copy()
    abdpain_consults = pd.merge(consults,abdpain_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    abdpain_consults = abdpain_consults[(abdpain_consults.requestDateTime >= abdpain_consults.EDStartDateTime) & 
                       (abdpain_consults.requestDateTime <= abdpain_consults.EDEndDateTime)].copy()
    print(dizzy_consults.shape,abdpain_consults.shape)
    return dizzy_consults,abdpain_consults

# count number of consults for each patient
def make_nconsults(consults):
    nconsults = consults.groupby(['PtSSN','EDStartDateTime'])['ConStopCodeName'].agg('count').reset_index().rename(columns={'ConStopCodeName':'nconsults'})
    
    return nconsults

def topN_consult_counts(consults,demo_df,N):
    # look at which departments are being consulted (pick top N)
    consult_codes = consults.groupby('ConStopCodeName')['PtSSN'].agg('count').sort_values(ascending=False).reset_index()
    ccodes = consult_codes.iloc[:N].ConStopCodeName
    
    # count up for each patient how many of each of these codes occurs
    # filter dizzy_consults to only include the top three codes
    consults_filtered = consults[consults.ConStopCodeName.isin(ccodes)]
    tmp = consults_filtered.groupby(['PtSSN','ConStopCodeName'])['PatientSID'].agg('count').reset_index().rename(columns={'PatientSID':'consult_count'})
   
    # pivot table and convert ConStopCodeName to another axis
    topNconsult_counts = pd.pivot_table(tmp,index='PtSSN',columns='ConStopCodeName',values='consult_count').fillna(0)
    
    # patients not in consults should be assigned zero counts.
    result = pd.merge(demo_df[['PtSSN']],topNconsult_counts,on='PtSSN',how='left').fillna(0).astype(int)
    return result
   

#########################################################################################################################
# Imaging data
#########################################################################################################################

def fill_zero_df(df,cohort_df):
    return pd.merge(cohort_df[['PtSSN','EDStartDateTime']],df,on=['PtSSN','EDStartDateTime'],how='left').fillna(0)


def is_img_abnormal(radiology_code):
    if radiology_code:
        return ('abnormal' in radiology_code.lower()) | ('malignan' in radiology_code.lower()) 
    else:
        return False
    
def is_ct_with_contrast(cptname):
    return ('w/contrast' in cptname.lower()) | ('w/dye' in cptname.lower())

def is_ct_head(cptname):
    return ('CT HEAD' in cptname) | ('CT ANGIOGRAPHY HEAD' in cptname)

def is_ct_abd(cptname):
    return ('CT ABD' in cptname) | ('CT NECK' in cptname) | ('CT PELVIS' in cptname) | ('CT ANGIOGRAPHY CHEST' in cptname)

def is_xray_abd(cptname):
    return ('X-RAY' in cptname)

# mode_name can be ct, xray, mri
# mode_fn can be functions for ct, xray, mri
# images can be dizzy images or abdpain images

def get_images_mode(images,mode_name,mode_fn,abnormal_fn,contrast_fn=None):
    images_mode = images[images.CPTName.apply(mode_fn)]
    images_mode_count = images_mode.groupby(['PtSSN','EDStartDateTime'])['PatientSID'].agg('count').reset_index().rename(columns={'PatientSID':mode_name+'_count'})
    if contrast_fn:
        images_mode_contrast = images_mode[images_mode.CPTName.apply(contrast_fn)]
        images_mode_contrast_count = images_mode_contrast.groupby(['PtSSN','EDStartDateTime'])['PatientSID'].agg('count').reset_index().rename(columns={'PatientSID':mode_name+'_contrast_count'})
    else:
        images_mode_contrast_count=None
    images_mode_abnormal = images_mode[images_mode.RadiologyDiagnosticCode.apply(abnormal_fn)]
    images_mode_abnormal_count = images_mode_abnormal.groupby(['PtSSN','EDStartDateTime'])['PatientSID'].agg('count').reset_index().rename(columns={'PatientSID':mode_name+'_abnormal_count'})
    return images_mode_count,images_mode_contrast_count,images_mode_abnormal_count

def is_xray(cptname):
    return 'X-RAY' in cptname

def is_mri(cptname):
    return 'MRI ' in cptname

def is_us(cptname):
    return 'US ' in cptname

def separate_cohorts_images(images,dizzy_cohort_df,abdpain_cohort_df):

    images.rename(columns={'PatientSSN':'PtSSN'},inplace=True)
    images.PtSSN = images.PtSSN.astype('int64')

    # restrict images to dizzy cohort
    dizzy_img = pd.merge(images,dizzy_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    # restrict images to dizzy cohort within the ED Start and End Times
    dizzy_img = dizzy_img[(dizzy_img.ExamDateTime >= dizzy_img.EDStartDateTime) & (dizzy_img.ExamDateTime <= dizzy_img.EDEndDateTime)].copy()

    # restrict images to abpain cohort
    abdpain_img = pd.merge(images,abdpain_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    # restrict images to abpain cohort within the ED Start and End Times
    abdpain_img = abdpain_img[(abdpain_img.ExamDateTime >= abdpain_img.EDStartDateTime) & (abdpain_img.ExamDateTime <= abdpain_img.EDEndDateTime)].copy()
    
    # clean up records with no CPTName
    dizzy_img = dizzy_img.dropna(subset = ['CPTName'])
    abdpain_img = abdpain_img.dropna(subset=['CPTName'])
    
    
    # CT images
    dizzy_images_ct_count, dizzy_images_ct_contrast_count, dizzy_images_ct_abnormal_count = get_images_mode(dizzy_img,'ct',is_ct_head,is_img_abnormal,is_ct_with_contrast)
    abdpain_images_ct_count, abdpain_images_ct_contrast_count, abdpain_images_ct_abnormal_count = get_images_mode(abdpain_img,'ct',is_ct_abd,is_img_abnormal,is_ct_with_contrast)

    #Xray images
    dizzy_images_xr_count, dizzy_images_xr_contrast_count, dizzy_images_xr_abnormal_count = get_images_mode(dizzy_img,'xr',is_xray,is_img_abnormal,None)
    abdpain_images_xr_count, abdpain_images_xr_contrast_count, abdpain_images_xr_abnormal_count = get_images_mode(abdpain_img,'xr',is_xray,is_img_abnormal,None)

    #MRI images
    dizzy_images_mri_count, dizzy_images_mri_contrast_count, dizzy_images_mri_abnormal_count = get_images_mode(dizzy_img,'mri',is_mri,is_img_abnormal,is_ct_with_contrast)
    abdpain_images_mri_count, abdpain_images_mri_contrast_count, abdpain_images_mri_abnormal_count = get_images_mode(abdpain_img,'mri',is_mri,is_img_abnormal,is_ct_with_contrast)

    #US images for abdpain only

    abdpain_images_us_count, abdpain_images_us_contrast_count, abdpain_images_us_abnormal_count = get_images_mode(abdpain_img,'us',is_us,is_img_abnormal,None)
    
    # fix the zero counts
    # CT
    dizzy_images_ct_count = fill_zero_df(dizzy_images_ct_count,dizzy_cohort_df)
    dizzy_images_ct_contrast_count = fill_zero_df(dizzy_images_ct_contrast_count,dizzy_cohort_df)
    dizzy_images_ct_abnormal_count = fill_zero_df(dizzy_images_ct_abnormal_count,dizzy_cohort_df)

    abdpain_images_ct_count = fill_zero_df(abdpain_images_ct_count,abdpain_cohort_df)
    abdpain_images_ct_contrast_count = fill_zero_df(abdpain_images_ct_contrast_count,abdpain_cohort_df)
    abdpain_images_ct_abnormal_count = fill_zero_df(abdpain_images_ct_abnormal_count,abdpain_cohort_df)

    #XRay
    dizzy_images_xr_count = fill_zero_df(dizzy_images_xr_count,dizzy_cohort_df)
    dizzy_images_xr_abnormal_count = fill_zero_df(dizzy_images_xr_abnormal_count,dizzy_cohort_df)

    abdpain_images_xr_count = fill_zero_df(abdpain_images_xr_count,abdpain_cohort_df)
    abdpain_images_xr_abnormal_count = fill_zero_df(abdpain_images_xr_abnormal_count,abdpain_cohort_df)

    #MRI
    dizzy_images_mri_count = fill_zero_df(dizzy_images_mri_count,dizzy_cohort_df)
    dizzy_images_mri_abnormal_count = fill_zero_df(dizzy_images_mri_abnormal_count,dizzy_cohort_df)

    abdpain_images_mri_count = fill_zero_df(abdpain_images_mri_count,abdpain_cohort_df)
    abdpain_images_mri_abnormal_count = fill_zero_df(abdpain_images_mri_abnormal_count,abdpain_cohort_df)

    #US

    abdpain_images_us_count = fill_zero_df(abdpain_images_us_count,abdpain_cohort_df)
    abdpain_images_us_abnormal_count = fill_zero_df(abdpain_images_us_abnormal_count,abdpain_cohort_df)

    # combine into CT, XRay, MRI for dizzy and CT,XRay, MRI, US for abdpain
    
    tmp_dizzy = pd.merge(dizzy_images_ct_count,dizzy_images_ct_contrast_count,on=['PtSSN','EDStartDateTime'])
    tmp_dizzy = pd.merge(tmp_dizzy,dizzy_images_ct_abnormal_count,on=['PtSSN','EDStartDateTime'])
    tmp_dizzy = pd.merge(tmp_dizzy,dizzy_images_xr_count,on=['PtSSN','EDStartDateTime'])
    tmp_dizzy = pd.merge(tmp_dizzy,dizzy_images_xr_abnormal_count,on=['PtSSN','EDStartDateTime'])
    tmp_dizzy = pd.merge(tmp_dizzy,dizzy_images_mri_count,on=['PtSSN','EDStartDateTime'])
    tmp_dizzy = pd.merge(tmp_dizzy,dizzy_images_mri_abnormal_count,on=['PtSSN','EDStartDateTime'])
    
    tmp_abdpain = pd.merge(abdpain_images_ct_count,abdpain_images_ct_contrast_count,on=['PtSSN','EDStartDateTime'])
    tmp_abdpain = pd.merge(tmp_abdpain,abdpain_images_ct_abnormal_count,on=['PtSSN','EDStartDateTime'])
    tmp_abdpain = pd.merge(tmp_abdpain,abdpain_images_xr_count,on=['PtSSN','EDStartDateTime'])
    tmp_abdpain = pd.merge(tmp_abdpain,abdpain_images_xr_abnormal_count,on=['PtSSN','EDStartDateTime'])
    tmp_abdpain = pd.merge(tmp_abdpain,abdpain_images_mri_count,on=['PtSSN','EDStartDateTime'])
    tmp_abdpain = pd.merge(tmp_abdpain,abdpain_images_mri_abnormal_count,on=['PtSSN','EDStartDateTime'])
    tmp_abdpain = pd.merge(tmp_abdpain,abdpain_images_us_count,on=['PtSSN','EDStartDateTime'])
    tmp_abdpain = pd.merge(tmp_abdpain,abdpain_images_us_abnormal_count,on=['PtSSN','EDStartDateTime'])
    
    
    return tmp_dizzy.drop_duplicates(), tmp_abdpain.drop_duplicates()



#########################################################################################################################
# Labs data
#########################################################################################################################

def matches_WBC_loinc(loinc):
    return loinc in ['6690-2']

def matches_albumin_loinc(loinc):
    return loinc == '1751-7'

def matches_glucose_loinc(loinc):
    return loinc in ['2345-7', '27353-2', '2340-8', '41651-1', '32016-8', '2339-0']

def matches_sodium_loinc(loinc):
    return loinc in ['2951-2','32717-1','2947-0','39791-9']

def matches_calcium_loinc(loinc):
    return loinc in ['17861-6','2000-8']

def matches_potassium_loinc(loinc):
    return loinc in ['2823-3', '32713-0', '6298-4', '39789-3']

def matches_chloride_loinc(loinc):
    return loinc in ['2075-0', '41650-3', '2069-3', '2072-7', '41649-5']

def matches_lact_loinc(loinc):
    return loinc in ['2524-7','14118-4','2578-9','2519-7','19240-1','30241-4','32693-4']

def matches_bun_loinc(loinc):
    return loinc == '3094-0'

def matches_creat_str(s):
    '''returns True if s contains creatinine, but not ratio, egfr, dau, or ur(ine). 24-hr might fall through. dau is urine.'''
    return (s.lower().find('creatinine')>=0) and not (s.lower().find('ratio')>=0) and not (s.lower().find('egfr')>=0) and not (s.lower().find('ur')>=0) and not (s.lower().find('dau')>=0)

def matches_creat_loinc(loinc):
    return loinc in ['2160-0','21232-4','38483-4']

def matches_CO2_loinc(loinc):
    return loinc == '2028-9'

def matches_troponin_loinc(loinc):
    return loinc in ['10839-9', '42757-5']

def matches_ast_loinc(loinc):
    return loinc == '1920-8'

def matches_alt_loinc(loinc):
    return loinc in ['1742-6', '1743-4', '1744-2']

def matches_alkphos_loinc(loinc):
    return loinc == '6768-6'

def matches_lipase_loinc(loinc):
    return loinc == '3040-3'

def matches_amylase_loinc(loinc):
    return loinc == '1798-8'

def matches_hgb_loinc(loinc):
    return loinc in ['718-7', '30313-1']

def separate_cohorts_labs(labs,dizzy_cohort_df,abdpain_cohort_df):

    labs.rename(columns={'PatientSSN':'PtSSN'},inplace=True)
    labs.PtSSN = labs.PtSSN.astype('int64')

    # restrict labs to dizzy cohort
    dizzy_labs = pd.merge(labs,dizzy_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    # restrict labs to dizzy cohort within the ED Start and End Times
    dizzy_labs = dizzy_labs[(dizzy_labs.LabChemSpecimenDateTime >= dizzy_labs.EDStartDateTime) & (dizzy_labs.LabChemSpecimenDateTime <= dizzy_labs.EDEndDateTime)].copy()

    # restrict labs to abpain cohort
    abdpain_labs = pd.merge(labs,abdpain_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    # restrict labs to abpain cohort within the ED Start and End Times
    abdpain_labs = abdpain_labs[(abdpain_labs.LabChemSpecimenDateTime >= abdpain_labs.EDStartDateTime) & (abdpain_labs.LabChemSpecimenDateTime <= abdpain_labs.EDEndDateTime)].copy()

    return dizzy_labs, abdpain_labs


def get_labs_data(labs,labname,labfn,cohort_df):
    # get subset of labs that match labfn
    lab_subset = labs[labs.LOINC.apply(labfn)]
    # group by PtSSN, get count, min, max
    lab_subset_data = lab_subset.groupby('PtSSN')['LabChemResultNumericValue'].agg(['count','min','max']).reset_index()
    lab_subset_data.columns=['PtSSN',labname+'_count',labname+'_min',labname+'_max']
    lab_subset_abnormal_count = lab_subset.groupby('PtSSN')['Abnormal'].agg('count').reset_index().rename(columns={'Abnormal':labname+'_abnormal_count'})
    
    # combine the count, min, max, and abnormal_count into one dataset
    tmp = pd.merge(lab_subset_data,lab_subset_abnormal_count,on='PtSSN')
    pts = pd.DataFrame(cohort_df['PtSSN'].unique(),columns=['PtSSN'])
    tmp1 = pd.merge(pts,tmp,on='PtSSN',how='left')
    # convert NaN count fields to zero
    tmp1[labname+'_count'].fillna(0,inplace=True)
    tmp1[labname+'_abnormal_count'].fillna(0,inplace=True)
    return tmp1

    #return lab_subset_data, lab_subset_abnormal_count

# 
def is_lab_relevant(labs,labname,labfn,labels_df):
    lab_subset_data, lab_subset_abnormal_count = get_labs_data(labs,labname,labfn)
    lab_subset_data_labeled = pd.merge(lab_subset_data,labels_df,on='PtSSN')
    #print(labname,lab_subset_data.shape[0],lab_subset_data_labeled.shape[0])
    lab_cols = ttest_fields(lab_subset_data_labeled,lab_subset_data.columns[1:],len(lab_subset_data.columns[1:])*['c'])  
    lab_subset_abnormal_count_labeled = pd.merge(lab_subset_abnormal_count,labels_df,on='PtSSN')
    tmp_cols = lab_subset_abnormal_count.columns[1:]
    ab_lab_cols = ttest_fields(lab_subset_abnormal_count_labeled,tmp_cols,len(tmp_cols)*['c'])
    return lab_cols, ab_lab_cols, lab_subset_data, lab_subset_abnormal_count

#########################################################################################################################
# Risk factors data
#########################################################################################################################
ICD9_FILEPATH = "P:\ORD_Singh_201911038D\Maxxxx\ICD\p_Refined_SPADE_RiskFactors_riskfactor_ICD9codes.csv"
ICD10_FILEPATH = "P:\ORD_Singh_201911038D\Maxxxx\ICD\p_Refined_SPADE_RiskFactors_riskfactor_ICD10codes.csv"

# do some standardization of terminology in the two ICD tables
terminology_map = {'Hx of stroke/TIA':'Hx of stroke or TIA',
                      ' Hx of atrial fibrillation':'Atrial fibrillation',
                      'Hyperlipdemia':'Hyperlipidemia',
                      'Hx of cerebral aneurysm':'Hx aneurysm'}

def standardize_term(description):
    if description in terminology_map:
        return terminology_map[description]
    else:
        return description                    

def get_dizzy_risk_factor_codes():
    icd9s = pd.read_csv(ICD9_FILEPATH)[['ICD-9-CM CODE','Description']].rename(columns={'ICD-9-CM CODE':'ICD'})
    icd10s = pd.read_csv(ICD10_FILEPATH)[['ICD-10-CM Code','Risk factors description']].rename(columns={'ICD-10-CM Code':'ICD','Risk factors description':'Description'})
    all_codes = pd.concat([icd9s,icd10s])
    all_codes.Description = all_codes.Description.apply(lambda x: standardize_term(x))
    return all_codes
    
def separate_cohorts_icds(icds,dizzy_cohort_df,abdpain_cohort_df):

    icds.rename(columns={'PatientSSN':'PtSSN'},inplace=True)
    icds.PtSSN = icds.PtSSN.astype('int64')

    # restrict icds to dizzy cohort
    dizzy_icds = pd.merge(icds,dizzy_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    # restrict icds to dizzy cohort to before ED Start 
    dizzy_icds = dizzy_icds[(dizzy_icds.DiagDateTime <= dizzy_icds.EDEndDateTime)].copy()

    # restrict icds to abpain cohort
    abdpain_icds = pd.merge(icds,abdpain_cohort_df[['PtSSN','EDStartDateTime','EDEndDateTime']],on='PtSSN')
    # restrict icds to abpain cohort before ED Start 
    abdpain_icds = abdpain_icds[(abdpain_icds.DiagDateTime <= abdpain_icds.EDEndDateTime)].copy()

    return dizzy_icds, abdpain_icds

# filter to the dizzy codes
def get_dizzy_rf(dizzy_icds,dizzy_cohort_df):
    all_codes = get_dizzy_risk_factor_codes()
    dizzy_rf_icds = pd.merge(dizzy_icds[dizzy_icds.IsPrincipalDiag==1],all_codes,on='ICD')
    tmp = dizzy_rf_icds.groupby(['PtSSN','Description'])['ICD'].agg('count').reset_index()
    dizzy_rf_df = pd.pivot_table(tmp,index='PtSSN',columns='Description',values='ICD').fillna(0)
    
    dizzy_pts = pd.DataFrame(dizzy_cohort_df['PtSSN'].unique(),columns=['PtSSN'])
    dizzy_rf_df_all = pd.merge(dizzy_pts,dizzy_rf_df,on='PtSSN',how='left').fillna(0)
    
    return dizzy_rf_df_all

# filter the abdpain risk factors

def get_abdpain_rf(abdpain_icds,abdpain_cohort_df):
    abdpain_cirrhosis = abdpain_icds[abdpain_icds.ICD.str.match(r"(K70.3)|(K71.7)|(K74.[3|4|5|6]) ")].copy()
    abdpain_cirrhosis['risk_factor'] = 'cirrhosis'
    abdpain_gallbladder = abdpain_icds[abdpain_icds.ICD.str.match(r"(K80.[0|1|3|4|6])|K81")].copy()
    abdpain_gallbladder['risk_factor'] = 'cholecystisis'
    abdpain_cholelithiasis = abdpain_icds[abdpain_icds.ICD.str.match(r"(K80.[2|5|7|8])")].copy()
    abdpain_cholelithiasis['risk_factor'] = 'cholelithiasis'
    abdpain_appendicitis = abdpain_icds[abdpain_icds.ICD.str.match(r"K3[5-8].*")].copy()
    abdpain_appendicitis['risk_factor'] = 'appendicitis'
    abdpain_diverticulitis = abdpain_icds[abdpain_icds.ICD.str.match(r"(K57.0[0-1])|(K57.1[2-3]) | (K57.2[1-2]) | (K57.3[2-3]) | (K57.4[0-1]) | (K57.5[2-3]) | (K57.8[1-2]) | (K57.9[2-3])")].copy()
    abdpain_diverticulitis['risk_factor'] = 'diverticulitis'
    abdpain_diverticulosis = abdpain_icds[abdpain_icds.ICD.str.match(r"(K57.1[0-1])|(K57.3[0-1]) | (K57.5[0-1]) | K57.9[0-1]")].copy()
    abdpain_diverticulosis['risk_factor'] = 'diverticulosis'
    abdpain_ib = abdpain_icds[abdpain_icds.ICD.str.match(r"K5[0-2].*")].copy()
    abdpain_ib['risk_factor'] = 'IB'
    abdpain_pancreatitis = abdpain_icds[abdpain_icds.ICD.str.match(r"(K85)|(K86.0)|(K86.1)")].copy()
    abdpain_pancreatitis['risk_factor'] = 'pancreatitis'
    abdpain_rf_icds = pd.concat([abdpain_cirrhosis,abdpain_gallbladder,abdpain_cholelithiasis,abdpain_appendicitis,abdpain_diverticulitis,
                             abdpain_diverticulosis,abdpain_ib,abdpain_pancreatitis])
    abdpain_rf_icds = abdpain_rf_icds[abdpain_rf_icds.IsPrincipalDiag==1]
    tmp = abdpain_rf_icds.groupby(['PtSSN','risk_factor'])['ICD'].agg('count').reset_index()
    abdpain_rf_df = pd.pivot_table(tmp,index='PtSSN',columns='risk_factor',values='ICD').fillna(0)
    abdpain_pts = pd.DataFrame(abdpain_cohort_df['PtSSN'].unique(),columns=['PtSSN'])
    abdpain_rf_df_all = pd.merge(abdpain_rf_df,abdpain_pts,on='PtSSN',how='right').fillna(0)
    return abdpain_rf_df_all

# Modeling code
from sklearn.model_selection import GridSearchCV

def tune_model(X,y,nfolds=5):
    alphas = np.logspace(-4,2,30)
    tuned_parameters = [{'C':alphas}]
    

    lr = LogisticRegression(max_iter=10000,penalty='l1',solver='saga')
    

    clf = GridSearchCV(lr,tuned_parameters,cv=nfolds,refit=False,scoring='f1_macro')
    clf.fit(X,y)
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']

    # plot everything
    plt.semilogx(alphas,scores)
    std_error = scores_std/np.sqrt(nfolds)

    plt.semilogx(alphas,scores+std_error,'b--')
    plt.semilogx(alphas,scores-std_error,'b--')
    plt.fill_between(alphas,scores + std_error,scores-std_error,alpha=0.2)
    plt.ylabel('CV score +/- std error')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores),linestyle='--',color='r')
    plt.xlim([alphas[0],alphas[-1]])
    return clf



def build_L1_model(X,y,C):
    print(X.shape,y.shape)
    clf = LogisticRegression(C=C,max_iter=10000,penalty='l1',solver='saga')
    scores = cross_val_score(clf,X,y,cv=5,scoring='roc_auc')
    print('AUC = ',scores.mean(), scores.std())
    scores = cross_val_score(clf,X,y,cv=5,scoring='f1_macro')
    print('F1 = ',scores.mean(), scores.std())
    scores = cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    print('Accuracy = ', scores.mean(), scores.std())

    clf = LogisticRegression(C=C,max_iter=10000,penalty='l1',solver='saga')
    clf.fit(X,y)
    print('Confusion matrix:')
    ypred = clf.predict(X)
    print(metrics.confusion_matrix(y,ypred))
    select_feats = list(X.columns[np.where(clf.coef_ != 0)[1]])
    print(len(select_feats), ' features chosen by L1 model.')
    return clf, select_feats

def visualize_model(clf,select_feats):
    coef_df = pd.concat([pd.DataFrame(select_feats),pd.DataFrame(clf.coef_[clf.coef_ !=0])],axis=1)
    coef_df.columns=['Name','Coef_value']
    coef_df = coef_df.sort_values(by='Coef_value')
    sns.barplot(data=coef_df,y='Name',x='Coef_value')
