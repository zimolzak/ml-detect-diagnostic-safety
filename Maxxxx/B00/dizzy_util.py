import pyodbc
import pandas as pd  # sure takes a long time
import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from collections import defaultdict
from feature_util import *
import sklearn
from sklearn.preprocessing import StandardScaler



def retrieveLabels():
    dizziness_df = pd.read_csv(open("Viral_data/ML4_HighRisk_Dizziness/labels-Dizziness.csv", 'r', errors="ignore"))
    dizziness_df.dropna(subset=["DxErrorER", "PtSSN"], inplace=True)
    dizziness_df = dizziness_df.astype("string")
    dizziness_df["PtSSN"] = dizziness_df["PtSSN"].astype(int)
    display(dizziness_df[["DxErrorER", "DxErrorERCoded"]])
    label_map = dict()
    for index, row in dizziness_df.iterrows():
        if row["PtSSN"] in label_map:
            print("duplicate label???")
        label_map[row["PtSSN"]] = row["DxErrorERCoded"]
    return dizziness_df, label_map

def convertLabelMap(label_map):
    df = pd.DataFrame.from_dict(label_map, orient="index", columns=['Label'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index':'PatientSSN'})
    return df

def convertBinaryLabel(label_df):
#     label_dict = {"MOD": 1, "PMOD": 1, "NoMOD": 0, "CodingEr": None}
    label_dict = {"MOD": 1, "PMOD": 1, "NoMOD": 0, "CodingEr": 2}
    df = label_df.copy()
    df["Label"] = df["Label"].map(label_dict)
    return df

def makeDataset(feature_vec, label_df, key="PatientSSN"):
    df = feature_vec.merge(convertBinaryLabel(label_df), how="outer", left_on=key, right_on=key)
    # labeled_df = labeled_df.fillna("Unlabeled")
    labeled_df = df[~pd.isnull(df['Label'])].copy()
    labeled_df = labeled_df[labeled_df["Label"] != 2] # filter out coding errors
    unlabeled_df = df[pd.isnull(df['Label'])].copy()
    X_cols = list(feature_vec.columns)
    X_cols.remove(key)
    y_col = 'Label'
    return labeled_df, unlabeled_df, X_cols, y_col

def colorUMAPwithLabels(label_map, ids, embedding):
#     color_map = {"MOD": "red","PMOD":"purple", "NoMOD":"blue", "CodingEr":"green"}
#     colors = [ color_map[label_map[id]] if id in label_map else "white" for id in ids]
#     fig = plt.figure(figsize=(12,8), dpi=100)
#     plt.style.use("dark_background")
#     plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.5)
#     plt.show()
    classes = ["MOD", "PMOD", "NoMOD", "CodingEr", "Unkown"]
    color_map = ["red", "purple", "blue", "green", "white"]
    class_map = {"MOD": 0, "PMOD": 1, "NoMOD": 2, "CodingEr": 3, "Unkown": 4}
    colors = [ class_map[label_map[id]] if id in label_map else 4 for id in ids]
    fig = plt.figure(figsize=(12,8), dpi=100)
    plt.style.use("dark_background")
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=ListedColormap(color_map), alpha=0.5)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.show()
    
def UMAPscalePlot(reducer, feature_vec, label_map):
    raise Exception("Deprecated!!")
    X = feature_vec.iloc[:,1:].values.astype('float') # cut out only data we wish to use
    scaler = StandardScaler()
    sX = scaler.fit_transform(X) # scale appropriately
    plt.style.use("dark_background")
    embedding = reducer.fit_transform(sX)
    
    colorUMAPwithLabels(label_map, feature_vec.PatientSSN, embedding)
    return embedding

def UMAPPlot(reducer, feature_vec, label_map):
    embedding = reducer.fit_transform(feature_vec.iloc[:,1:])
    
    colorUMAPwithLabels(label_map, feature_vec.PatientSSN, embedding)
    return embedding

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
        codes[NORM_MAP[name]].append(str(code))
    
    icd10s = pd.read_csv(ICD10_FILEPATH)
    # assert set(icd10s["Risk factors description"]) == set(NORM_MAP.values())
    
    for code, name in zip(icd10s["ICD-10-CM Code"], icd10s["Risk factors description"]):
        codes[name].append(str(code))
    
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

class ICDFeature(Feature):
    def __init__(self, vec):
        Feature.__init__(self, vec, ICD_PATIENT_ID)
        
    def normalize(self, vec):
        # vec.iloc[:,1:] = StandardScaler().fit_transform(vec.iloc[:,1:])
        vec.iloc[:,1:] = vec.iloc[:,1:].clip(upper=2) / 2
        return vec
    


def makeICDFeatures(icd_df):
    return ICDFeature(makeICDFeatureVector(icd_df))


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
        if not df.empty:
            temp = df.copy()
            temp["index_time"] = temp.apply(lambda r: index_times[r[DRUG_PATIENT_ID]], axis=1)
            temp["temp"] = (temp["index_time"] - temp["DispensedDate"]).dt.days
            temp[name] = temp.apply(lambda r: calcWindowDaysSupply(r["temp"], r["DaysSupply"], considered_window), axis=1)
            temp.drop(temp.columns.difference([DRUG_PATIENT_ID, name]), 1, inplace=True)
            vec = vec.merge(temp.groupby(DRUG_PATIENT_ID).sum(), how="outer", left_on=DRUG_PATIENT_ID, right_on=DRUG_PATIENT_ID)
        else:
            vec[name] = 0
    vec.fillna(0, inplace=True)
    return vec


class OutpatFeature(Feature):
    def __init__(self, vec):
        super().__init__(vec, DRUG_PATIENT_ID)
        
    def normalize(self, vec):
        # vec.iloc[:,1:] = StandardScaler().fit_transform(vec.iloc[:,1:])
        vec.iloc[:,1:] = vec.iloc[:,1:].clip(upper=1)
        return vec


def makeOutpatWindowFeatures(outpat_df, index_times, considered_window=180):
    return OutpatFeature(makeRxOutpatTimeWindowVec(outpat_df, index_times, considered_window))


##########################################################################################################
###########################################    Consult   #################################################
##########################################################################################################

NEURO_CONSULT_CODES = [315.0, 325.0]
CONSULT_ID = "PatientSSN"

def makeConsultVec(consult_df, index_times, window=36):
    neuro_df = filterDFByCodeSet(consult_df, "ConStopCode", NEURO_CONSULT_CODES).copy()
    convertDatetime(neuro_df, ["requestDateTime"])
    vec = pd.DataFrame({CONSULT_ID:neuro_df[CONSULT_ID]})
    
    neuro_df["index_time"] = neuro_df.apply(lambda r: index_times[r[CONSULT_ID]], axis=1)
    neuro_df["temp"] = (neuro_df["index_time"] - neuro_df["requestDateTime"]).dt.total_seconds() / 3600
    vec["HasConsult"] = neuro_df.apply(lambda r: 1 if abs(r["temp"]) < window else 0, axis=1)
    consult_vec = vec.groupby(CONSULT_ID).sum()
    consult_vec.HasConsult = consult_vec.HasConsult.clip(upper=1)
    return consult_vec

def makeConsultFeature(consult_df, index_times, window=36):
    return Feature(makeConsultVec(consult_df, index_times, window), CONSULT_ID)

##########################################################################################################
###########################################     Rad     ##################################################
##########################################################################################################

CT_CPT_CODES = ['70450', '70460', '70470']
MRI_CPT_CODES = ['70550', '70551', '70552', '70553']

RAD_PATIENT_ID = "PatientSSN"

def makeRadCTVec(rad_df, index_times, window=36):
    ct_df = filterDFByCodeSet(rad_df, "CPTCode", CT_CPT_CODES).copy()
    convertDatetime(ct_df, ["ExamDateTime"])
    vec = pd.DataFrame({CONSULT_ID:ct_df[RAD_PATIENT_ID]})
    
    ct_df["index_time"] = ct_df.apply(lambda r: index_times[r[RAD_PATIENT_ID]], axis=1)
    ct_df["temp"] = (ct_df["index_time"] - ct_df["ExamDateTime"]).dt.total_seconds() / 3600
    vec["HasCT"] = ct_df.apply(lambda r: 1 if abs(r["temp"]) < window else 0, axis=1)
    ct_vec = vec.groupby(RAD_PATIENT_ID).sum()
    ct_vec.HasCT = ct_vec.HasCT.clip(upper=1)
    return ct_vec

def makeRadMRIVec(rad_df, index_times, window=36):
    mri_df = filterDFByCodeSet(rad_df, "CPTCode", MRI_CPT_CODES).copy()
    convertDatetime(mri_df, ["ExamDateTime"])
    vec = pd.DataFrame({CONSULT_ID:mri_df[RAD_PATIENT_ID]})
    
    mri_df["index_time"] = mri_df.apply(lambda r: index_times[r[RAD_PATIENT_ID]], axis=1)
    mri_df["temp"] = (mri_df["index_time"] - mri_df["ExamDateTime"]).dt.total_seconds() / 3600
    vec["HasMRI"] = mri_df.apply(lambda r: 1 if abs(r["temp"]) < window else 0, axis=1)
    mri_vec = vec.groupby(RAD_PATIENT_ID).sum()
    mri_vec.HasMRI = mri_vec.HasMRI.clip(upper=1)
    return mri_vec

def makeRadVec(rad_df, index_times, window=36):
    ct_vec = makeRadCTVec(rad_df, index_times, window)
    mri_vec = makeRadMRIVec(rad_df, index_times, window)
    vec = ct_vec.merge(mri_vec, how="outer", left_on=RAD_PATIENT_ID, right_on=RAD_PATIENT_ID)
    vec = vec.fillna(0)
    return vec

def makeRadFeature(rad_df, index_times, window=36):
    return Feature(makeRadVec(rad_df, index_times, window), RAD_PATIENT_ID)