import pandas as pd
import numpy as np
import pyodbc
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def getA00_Tables():
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=VHACDWRB03;DATABASE=ORD_Singh_201911038D")
    info_str = "select * from information_schema.tables where table_name like '_A00_RFSPADE%'"
    info_df = pd.read_sql(sql=info_str,con=conn)
    tables = {}
    for tname in info_df.TABLE_NAME:
        query_str = "select * from  " + str("Dflt.")+ str(tname)
        table_df = pd.read_sql(sql=query_str,con=conn)
        tables[tname.split('_')[-1]] = table_df
    return tables



def get_relevant_tables(tables):
    cohort = tables['cohort'].copy()
    ICDs = tables['ICD'].copy()
    RxOutpat = tables['RxOutpat'].copy()
    consult = tables['Consult'].copy()
    demog = tables['Demorgraphics'].copy()
    notes = tables['Note'].copy()
    rad = tables['Rad'].copy()
    
    # convert date time fields to pandas format
    datefields = ['EDStartDateTime', 'EDEndDateTime','AdmitDateTime', 'DischargeDateTime']
    for datef in datefields:
        cohort[datef] = pd.to_datetime(cohort[datef]) 
    ICDs.DiagDateTime = pd.to_datetime(ICDs.DiagDateTime)
    RxOutpat.DispensedDate = pd.to_datetime(RxOutpat.DispensedDate)
    consult.RequestDateTime = pd.to_datetime(consult.RequestDateTime)
    consult.CPRSOrderResultsDateTime = pd.to_datetime(consult.CPRSOrderResultsDateTime)
    demog.DOB = pd.to_datetime(demog.DOB)
    notes.EntryDateTime = pd.to_datetime(notes.EntryDateTime)
    rad.ExamDateTime = pd.to_datetime(rad.ExamDateTime)
    return cohort, ICDs, RxOutpat, consult, demog, notes, rad

# cribbed from Max

def filterDFByCodeSet(df, colName, codes):
    return df[df[colName].isin(codes)]




def extractLastVisitIndexDatetime(cohort_df):
    
    indexes = {}
    for patientId, dt in zip(cohort_df["patientSSN"], cohort_df["EDStartDateTime"]):
        if patientId in indexes:
            # TODO: revise
            indexes[patientId] = max(indexes[patientId], dt)
        else:
            indexes[patientId] = dt
    
    return indexes

def extractFirstVisitIndexDatetime(cohort_df):
    
    indexes = {}
    for patientId, dt in zip(cohort_df["patientSSN"], cohort_df["EDStartDateTime"]):
        if patientId in indexes:
            # TODO: revise
            indexes[patientId] = min(indexes[patientId], dt)
        else:
            indexes[patientId] = dt
            
    return indexes


def indexERVisitCount(cohort):
    visits = {}
    for pid in cohort.patientSSN:
    # what are index visits for patient?
        visits[pid] = cohort[cohort.patientSSN==pid].shape[0]
    return pd.DataFrame.from_dict(visits,orient='index',columns=['visit_count'])

def filterDFByTimes(df, idColName, tsColName, tsUpperBounds, tsLowerBounds=None):
    #convertDatetime(df, [tsColName])
    if tsLowerBounds is None:
        def filterDTfn(row):
            return row[tsColName] <= tsUpperBounds[row[idColName]]
    else:
        def filterDTfn(row):
            return row[tsColName] <= tsUpperBounds[row[idColName]] and row[tsColName] >= tsLowerBounds[row[idColName]]
    
    cond = df.apply(filterDTfn, axis=1)
    return df[cond]
    
   

def makeRxOutpatTimeWindowVec(outpat_df, index_times, considered_window=180):
    """
    outpat_df: the filtered data frame of RxOutpat events prior to the index times
    index_time: dictionary of times keyed by patient ids
    considered_window: size of window in days
    """
    outpat_df['index_time'] = outpat_df.apply(lambda r: index_times[r['PatientSSN']], axis=1)
    outpat_df['dispensed_date_from_index'] = (outpat_df["index_time"] - outpat_df["DispensedDate"]).dt.days
    # throw out all medicine records before 180 days from index visit
    outpat_df_filt = outpat_df[outpat_df.dispensed_date_from_index <= considered_window].copy()
    outpat_df_filt['Mod_quantity'] = outpat_df_filt.apply(lambda x: min(x['DaysSupply'],x['dispensed_date_from_index']),axis=1)
 
       
        
    return outpat_df_filt

ICD9_FILEPATH = "P:\ORD_Singh_201911038D\Maxxxx\ICD\p_Refined_SPADE_RiskFactors_riskfactor_ICD9codes.csv"
ICD10_FILEPATH = "P:\ORD_Singh_201911038D\Maxxxx\ICD\p_Refined_SPADE_RiskFactors_riskfactor_ICD10codes.csv"

ICD9_FILEPATH = "P:\ORD_Singh_201911038D\Maxxxx\ICD\p_Refined_SPADE_RiskFactors_riskfactor_ICD9codes.csv"
ICD10_FILEPATH = "P:\ORD_Singh_201911038D\Maxxxx\ICD\p_Refined_SPADE_RiskFactors_riskfactor_ICD10codes.csv"
icd9_codes = pd.read_csv(ICD9_FILEPATH)[['ICD-9-CM CODE','Description']]
icd9_codes.columns=['ICD','Description']
icd10_codes = pd.read_csv(ICD10_FILEPATH)[['ICD-10-CM Code','Risk factors description']]
icd10_codes.columns=['ICD','Description']

def isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False
    
def isnotfloat(x):
    return not isfloat(x)


def run_clustering_on_umap(mapper, num_clusters, cdata_df, columns):
    clust = KMeans(n_clusters=num_clusters)
    # clust = SpectralClustering(n_clusters=num_clusters)
    clust.fit(mapper.embedding_)
    plt.subplots(figsize=(10,10))
    for i in range(num_clusters):
        plt.scatter(mapper.embedding_[clust.labels_==i,0],mapper.embedding_[clust.labels_==i,1])
    plt.legend([i+1 for i in range(num_clusters)])
    for i in range(num_clusters):
        print("Cluster",i+1)
        print(cdata_df.iloc[np.where(clust.labels_==i)[0],:].shape[0])
        print()
    
    X = cdata_df.loc[:,columns].values.astype('float') # cut out only data we wish to use
    scaler = StandardScaler()
    sX = scaler.fit_transform(X) # scale appropriately
    
    # find average levels of each column for each cluster to tell them apart
    clusters = pd.DataFrame(data=clust.labels_, columns=["cluster"])
    cdata_scaled_df = pd.DataFrame(data=sX, columns=columns)
    clusters_data_df = pd.concat([cdata_scaled_df, clusters], axis=1)
    mean_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    median_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    std_clusters = np.zeros((clust.n_clusters, clusters_data_df.shape[1]-1))
    for i in range(clust.n_clusters):
        mean_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].mean()[:-1] # put in mean for each cluster
        median_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].median()[:-1] # put in median for each cluster
        std_clusters[i,:] = clusters_data_df[clusters_data_df["cluster"] == i].std()[:-1]/np.sqrt(clusters_data_df[clusters_data_df["cluster"] == i].count()[:-1]) # put in standard error for each cluster

    mean_clusters_df = pd.DataFrame(mean_clusters, columns=clusters_data_df.columns[:-1])
    median_clusters_df = pd.DataFrame(median_clusters, columns=clusters_data_df.columns[:-1])
    # mean_clusters_df.T.plot(kind='bar',figsize=(20,8), yerr=std_clusters) # bar chart, only use if necessary
    # plt.legend(["Cluster "+str(i+1) for i in range(4)])

    plt.figure(figsize=(20,3))
    plt.title('Means for each cluster')
    plt.imshow(mean_clusters_df,cmap='jet')
    plt.xticks(ticks=np.arange(mean_clusters_df.shape[1]), labels=mean_clusters_df.columns, rotation=90)
    plt.colorbar(shrink=0.5)
    plt.yticks(ticks=np.arange(mean_clusters_df.shape[0]), labels=["Cluster "+str(i+1) for i in range(mean_clusters_df.shape[0])])

    plt.figure(figsize=(20,3))
    plt.title('Medians for each cluster')
    plt.imshow(median_clusters_df,cmap='jet')
    plt.xticks(ticks=np.arange(median_clusters_df.shape[1]), labels=median_clusters_df.columns, rotation=90)
    plt.colorbar(shrink=0.5)
    plt.yticks(ticks=np.arange(median_clusters_df.shape[0]), labels=["Cluster "+str(i+1) for i in range(median_clusters_df.shape[0])])
    return clust
  
