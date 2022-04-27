import pyodbc
import pandas as pd  # sure takes a long time
import numpy as np
from collections import defaultdict

# deprecated: prioritize using precise ICD codes
# def extractICDFeature(icd_df, name, case=False, verbose=True):
#     df = icd_df.loc[icd_df['ICDDiagnosis'].str.contains(name, case=case)]
#     if verbose:
#         print("Patients", len(df.PatientSSN.unique()))
#         print("ICDs", len(df.ICD.unique()))
#         print("ICDDiagnosis'", len(df.ICDDiagnosis.unique()))

#         display(df)
    
#     return df


# extracts all datasets from database with selected prefix
def extractDataset(prefix):
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=VHACDWRB03;DATABASE=ORD_Singh_201911038D")

    info_df = pd.read_sql(sql="select * from information_schema.tables where table_name like '%{}%'".format(prefix), con=conn)
    display(info_df)
    # read all the tables into pandas tables
    tables = {}
    for tname in info_df.TABLE_NAME:
        query_str = "select * from  " + str("Dflt.")+ str(tname)
        table_df = pd.read_sql(sql=query_str,con=conn)
        tables[tname.split('_')[-1]] = table_df
    print(tables.keys())
    return tables


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
    

def mergeFeatures(feature_dict):
    feature_vec = None
    feature_key = None
    for k, v in feature_dict.items():
        if feature_vec is None:
            feature_key = k
            feature_vec = v
        else:
            feature_vec = feature_vec.merge(v, how="outer", left_on=feature_key, right_on=k)
            feature_vec.fillna(0, inplace=True)
    return feature_vec



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

def extractFirstVisitIndexDatetime(cohort_df):
    convertDatetime(cohort_df, COHORT_DATE_FIELDS)
    
    indexes = dict()
    for patientId, dt in zip(cohort_df["patientSSN"], cohort_df["EDStartDateTime"]):
        if patientId in indexes:
            # TODO: revise
            indexes[patientId] = min(indexes[patientId], dt)
        else:
            indexes[patientId] = dt
    
    return indexes



##########################################################################################################
###########################################    ICDs     ##################################################
##########################################################################################################

ICD9_FILEPATH = "P:\ORD_Singh_201911038D\Maxxxx\ICD\p_Refined_SPADE_RiskFactors_riskfactor_ICD9codes.csv"
ICD10_FILEPATH = "P:\ORD_Singh_201911038D\Maxxxx\ICD\p_Refined_SPADE_RiskFactors_riskfactor_ICD10codes.csv"

# Supported features:
HYPERTENSION = 'Hypertension'
HYPERLIPDEMIA = 'Hyperlipidemia'
DIABETES = 'Diabetes'
HX_STROKE_TIA = 'Hx of stroke or TIA'
HX_ATRIAL_FIBRILLATION = 'Atrial fibrillation'
CAD = 'Coronary artery disease (CAD)'
SMOKING = 'Smoking'
HX_ANEURYSM = 'Hx aneurysm'
OCCULSION_STENOSIS = 'Occlusion/Stenosis of cerebral or precerebral artery'

NORM_MAP = {
    ' Hx of atrial fibrillation' : HX_ANEURYSM,
    'Coronary artery disease (CAD)' : CAD,
    'Diabetes' : DIABETES,
    'Hx of cerebral aneurysm' : HX_ANEURYSM,
    'Hx of stroke/TIA' : HX_STROKE_TIA,
    'Hyperlipdemia' : HYPERLIPDEMIA,
    'Hypertension' : HYPERTENSION,
    'Occlusion/Stenosis of cerebral or precerebral artery' : OCCULSION_STENOSIS,
    'Smoking' : SMOKING}


ICD_PATIENT_ID = "PatientSSN"


def getICDCodes():
    codes = defaultdict(list)
    
    icd9s = pd.read_csv(ICD9_FILEPATH)
    # assert set(icd9s["Description"]) == set(NORM_MAP.keys())
    
    for code, name in zip(icd9s["ICD-9-CM CODE"], icd9s["Description"]):
        codes[NORM_MAP[name]].append(code)
    
    icd10s = pd.read_csv(ICD10_FILEPATH)
    # assert set(icd10s["Risk factors description"]) == set(NORM_MAP.values())
    
    for code, name in zip(icd10s["ICD-10-CM Code"], icd10s["Risk factors description"]):
        codes[name].append(code)
    
    return codes


def extractICDDataFrames(icd_df):
    icds = getICDCodes()
    dfs = dict()
    for name, codes in icds.items():
        dfs[name] = filterDFByCodeSet(icd_df, "ICD", codes)
    return dfs


def makeICDFeatureVector(icd_df):
    dfs = extractICDDataFrames(icd_df)
    vec = pd.DataFrame({ICD_PATIENT_ID:icd_df[ICD_PATIENT_ID].unique()})
    for name, df in dfs.items():
        temp = df.copy()
        temp.drop(temp.columns.difference([ICD_PATIENT_ID, "ICD"]), 1, inplace=True)
        temp.rename(columns={ICD_PATIENT_ID:ICD_PATIENT_ID, "ICD":name}, inplace=True)
        vec = vec.merge(temp.groupby(ICD_PATIENT_ID).count(), how="outer", left_on=ICD_PATIENT_ID, right_on=ICD_PATIENT_ID)
    vec = vec.fillna(0)
    return vec
    


##########################################################################################################
###########################################    Meds     ##################################################
##########################################################################################################

# TODO: read from csv instead?

# va_drug_classes = ['BL110','BL115','BL117','CV000','CV100','CV150','CV200','CV350','CV400','CV490','CV500',
#                   'CV701','CV702','CV704','CV709','CV800','CV805','CV806','HS500','HS501','HS502','HS503','HS509']
# va_super_category = ['anticoagulant','lytic','antiplatelet']+ ['antihypertensive']*4 +\
#                     ['cholesterol'] + ['antihypertensive']*10+['diabetes']*5

# DRUG_DF = pd.DataFrame(columns=['DrugClass','SuperCategory'])
# DRUG_DF['DrugClass'] = np.array(va_drug_classes)
# DRUG_DF['SuperCategory'] = np.array(va_super_category)

DRUG_FILEPATH = "P:\ORD_Singh_201911038D\ml-detect-diagnostic-safety-git-repo\short-list-drug-classes.csv"

DRUG_PATIENT_ID = "PatientSSN"

def getDrugCodesBySuperCategory():
    codes = defaultdict(list)
    drugs = pd.read_csv(DRUG_FILEPATH)
    for code, classname in zip(drugs["VA Class"], drugs["super category"]):
        codes[classname].append(code)
    return codes

def extractRxOutpatDataFrames(outpat_df):
    drug_codes = getDrugCodesBySuperCategory()
    dfs = dict()
    for category, codes in drug_codes.items():
        dfs[category] = filterDFByCodeSet(outpat_df, "DrugClassCode", codes)
    return dfs


def makeRxOutpatFeatureVector(outpat_df):
    dfs = extractRxOutpatDataFrames(outpat_df)
    vec = pd.DataFrame({DRUG_PATIENT_ID:outpat_df[DRUG_PATIENT_ID].unique()})
    for name, df in dfs.items():
        temp = df.copy()
        temp.drop(temp.columns.difference([DRUG_PATIENT_ID, "DrugClassCode"]), 1, inplace=True)
        temp.rename(columns={DRUG_PATIENT_ID:DRUG_PATIENT_ID, "DrugClassCode":name}, inplace=True)
        vec = vec.merge(temp.groupby(DRUG_PATIENT_ID).count(), how="outer", left_on=DRUG_PATIENT_ID, right_on=DRUG_PATIENT_ID)
    vec.fillna(0, inplace=True)
    return vec

def calcWindowDaysSupply(datediff, days_supply, considered_window):
    if datediff - considered_window < 0:
        return min(days_supply, datediff)
    else:
        return max(days_supply - (datediff - considered_window), 0)


def makeRxOutpatTimeWindowVec(outpat_df, index_times, considered_window=180):
    """
    outpat_df: the filtered data frame of RxOutpat events prior to the index times
    index_time: dictionary of times keyed by patient ids
    considered_window: size of window in days
    """
    dfs = extractRxOutpatDataFrames(outpat_df)
    vec = pd.DataFrame({DRUG_PATIENT_ID:outpat_df[DRUG_PATIENT_ID].unique()})
    for name, df in dfs.items():
        temp = df.copy()
        temp["index_time"] = temp.apply(lambda r: index_times[r[DRUG_PATIENT_ID]], axis=1)
        temp["temp"] = (temp["index_time"] - temp["DispensedDate"]).dt.days
        temp[name] = temp.apply(lambda r: calcWindowDaysSupply(r["temp"], r["DaysSupply"], considered_window), axis=1)
        temp.drop(temp.columns.difference([DRUG_PATIENT_ID, name]), 1, inplace=True)
        vec = vec.merge(temp.groupby(DRUG_PATIENT_ID).sum(), how="outer", left_on=DRUG_PATIENT_ID, right_on=DRUG_PATIENT_ID)
    vec.fillna(0, inplace=True)
    return vec


    