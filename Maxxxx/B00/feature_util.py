import pyodbc
import pandas as pd  # sure takes a long time
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# deprecated: prioritize using precise ICD codes
# def extractICDFeature(icd_df, name, case=False, verbose=True):
#     df = icd_df.loc[icd_df['ICDDiagnosis'].str.contains(name, case=case)]
#     if verbose:
#         print("Patients", len(df.PatientSSN.unique()))
#         print("ICDs", len(df.ICD.unique()))
#         print("ICDDiagnosis'", len(df.ICDDiagnosis.unique()))

#         display(df)
    
#     return df


def displayAll(df):
    pd.set_option('display.max_rows', len(df))
    display(df)
    pd.reset_option('display.max_rows')


# extracts all datasets from database with selected prefix
def extractDataset(prefix, excludeSet):
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=VHACDWRB03;DATABASE=ORD_Singh_201911038D")

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

def getPrimaryKeys(dataset):
    pkeys = dict()
    for k, v in dataset.items():
        pkeys[k] = list(v.columns)[[c.lower() for c in v.columns].index("patientssn")]
    return pkeys

def extractAndStandarizeCohort(dataset, subsetName):
    pkeys = getPrimaryKeys(dataset)
    cohort_key = pkeys["cohort"]
    cohort_subset = dataset["cohort"][dataset["cohort"]["TriggerType"] == subsetName]
    ids = pd.DataFrame({cohort_key: cohort_subset[cohort_key].unique()})
    datasubset = dict()
    for table, df in dataset.items():
        datasubset[table] = ids.merge(df, how="inner", left_on=cohort_key, right_on=pkeys[table])
        if pkeys[table] != cohort_key:
            datasubset[table] = datasubset[table].drop([cohort_key], axis=1)
        datasubset[table][pkeys[table]] = datasubset[table][pkeys[table]].astype(int)
    return datasubset


def filterDFByCodeSet(df, colName, codes):
    return df[df[colName].isin(codes)]


def convertDatetime(df, columns):
    for datef in columns:
        df[datef] = pd.to_datetime(df[datef])
        
def filterDFByTimes(df, idColName, tsColName, tsUpperBounds, tsLowerBounds=None):
    convertDatetime(df, [tsColName])
    if tsLowerBounds is None:
        def filterDTfn(row):
            return row[tsColName] <= tsUpperBounds[row[idColName]]
    else:
        def filterDTfn(row):
            return row[tsColName] <= tsUpperBounds[row[idColName]] and row[tsColName] >= tsLowerBounds[row[idColName]]
    
    cond = df.apply(filterDTfn, axis=1)
    return df[cond]
    



def normalizeFeatures(df, features):
    norm = df.copy()
    norm[features] = StandardScaler().fit_transform(df[features])
    return norm


class Feature:        
    # override when necessary
    def normalize(self, vec):
        return vec
    
    def __init__(self, vec, key="PatientSSN"):
        self.key = key
        self.vec = vec.copy()
        self.norm_vec = self.normalize(vec)
        
    def merge(self, feature, normalize=True):
        if normalize:
            feature_vec = self.norm_vec.merge(feature.norm_vec, how="outer", left_on=self.key, right_on=feature.key)
        else:
            feature_vec = self.vec.merge(feature.vec, how="outer", left_on=self.key, right_on=feature.key)
        if self.key != feature.key:
            feature_vec = feature_vec.drop([feature.key], axis=1)
        feature_vec.fillna(0, inplace=True)
        return Feature(feature_vec, self.key)


def mergeFeatures(features, normalize=True):
    """
    returns a pandas dataframe of feature vectors
    """
    feature = features[0]
    for i in range(1, len(features)):
        feature = feature.merge(features[i], normalize)
    return feature.vec
    
##########################################################################################################
###########################################    Cohort     ################################################
##########################################################################################################

# COHORT_DATE_FIELDS = ['EDStartDateTime', 'EDEndDateTime','AdmitDateTime', 'DischargeDateTime']
COHORT_DATE_FIELDS = ["EDStartDateTime", "EDEndDateTime", "AdmitDateTime", "B"]


def extractLastVisitIndexDatetime(cohort_df):
    convertDatetime(cohort_df, COHORT_DATE_FIELDS)
    
    indexes = dict()
    for patientId, dt in zip(cohort_df["patientSSN"], cohort_df["EDStartDateTime"]):
        if patientId in indexes:
            # TODO: revise
            indexes[patientId] = max(indexes[patientId], dt)
        else:
            indexes[patientId] = dt
    
    return indexes

def extractFirstDatetime(cohort_df, col_name):
    indexes = dict()
    for patientId, dt in zip(cohort_df["patientSSN"], cohort_df[col_name]):
        if patientId in indexes:
            # TODO: revise
            indexes[patientId] = min(indexes[patientId], dt)
        else:
            indexes[patientId] = dt
    
    return indexes

def extractFirstVisitIndexDatetime(cohort_df):
    convertDatetime(cohort_df, COHORT_DATE_FIELDS)
    return extractFirstDatetime(cohort_df, "EDStartDateTime")

def convertIdDictToDF(index_datetime, colName):
    df = pd.DataFrame.from_dict(index_datetime, orient="index", columns=[colName])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index':'PatientSSN'})
    return df

def convertIndexToDF(index_datetime):
    return convertIdDictToDF('IndexDate')

def makeIndexVec(cohort_df):
    start_df = convertIdDictToDF(extractFirstDatetime(cohort_df, "EDStartDateTime"), "EDStartDateTime")
    end_df = convertIdDictToDF(extractFirstDatetime(cohort_df, "EDEndDateTime"), "EDEndDateTime")
    admit_df = convertIdDictToDF(extractFirstDatetime(cohort_df, "AdmitDateTime"), "AdmitDateTime")
    time_df = start_df.merge(end_df.merge(admit_df))
    # in minutes
    time_df["ed_duration"] = (time_df["EDEndDateTime"] - time_df["EDStartDateTime"]).dt.total_seconds() / 60
    # in days
    time_df["ed_inp_delta"] = (time_df["AdmitDateTime"] - time_df["EDEndDateTime"]).dt.total_seconds() / (60 * 60 * 24)
    time_df["ed_inp_delta"] = time_df["ed_inp_delta"].clip(upper=30)
    time_df = time_df[["PatientSSN", "ed_duration", "ed_inp_delta"]]
    return time_df

class IndexFeature(Feature):
    def __init__(self, vec):
        Feature.__init__(self, vec, "PatientSSN")
        
    def normalize(self, vec):
        vec.iloc[:,1:] = MinMaxScaler().fit_transform(vec.iloc[:,1:])
        return vec

# TODO: clean up
class NormalIndexFeature(Feature):
    def __init__(self, vec):
        Feature.__init__(self, vec, "PatientSSN")
        
    def normalize(self, vec):
        vec["ed_duration"] = StandardScaler().fit_transform(vec["ed_duration"].values.reshape(-1,1))
        vec["ed_inp_delta"] = MinMaxScaler().fit_transform(vec["ed_duration"].values.reshape(-1,1))
        return vec
    
def makeIndexFeature(cohort_df, uniform=True):
    if uniform:
        return IndexFeature(makeIndexVec(cohort_df))
    else:
        return NormalIndexFeature(makeIndexVec(cohort_df))
    

def makeAgeVec(demo_df):
    df = demo_df.copy()
    df["Age"] = ((pd.to_datetime(demo_df["IndexDateTime"]) - pd.to_datetime(demo_df["DOB"])).dt.days / 365.25).astype(int)
    df = df[["patientSSN", "Age"]]
    df = df.rename(columns={'patientSSN':'PatientSSN'})
    return df

class AgeFeature(Feature):
    def __init__(self, vec):
        Feature.__init__(self, vec, "PatientSSN")
        
    def normalize(self, vec):
        vec.iloc[:,1:] = MinMaxScaler().fit_transform(vec.iloc[:,1:])
        return vec

class NormalAgeFeature(Feature):
    def __init__(self, vec):
        Feature.__init__(self, vec, "PatientSSN")
        
    def normalize(self, vec):
        vec.iloc[:,1:] = StandardScaler().fit_transform(vec.iloc[:,1:])
        return vec
    
def makeAgeFeature(demo_df, uniform=True):
    if uniform:
        return AgeFeature(makeAgeVec(demo_df))
    else:
        return NormalAgeFeature(makeAgeVec(demo_df))
    
##########################################################################################################
#######################################    Special Dates     #############################################
##########################################################################################################

def makeHolidayVec(index_datetime):
    index_df = convertIndexToDF(index_datetime)
    cal = calendar()
    holidays = cal.holidays(start=index_df['IndexDate'].min(), end=index_df['IndexDate'].max())
    index_df['Holiday'] = index_df['IndexDate'].isin(holidays).astype(int)
    return index_df
    
def makeWeekendVec(index_datetime):
    index_df = convertIndexToDF(index_datetime)
    index_df['IsWeekend'] = np.where((index_df['IndexDate'].dt.dayofweek) < 5, 0, 1)
    index_df = index_df.drop('IndexDate', 1)
    return index_df

def makeWeekendFeature(index_datetime):
    return Feature(makeWeekendVec(index_datetime))
    