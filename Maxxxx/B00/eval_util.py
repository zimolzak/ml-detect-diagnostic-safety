import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# machine learning
import sklearn
from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFECV, chi2, mutual_info_classif
# from xgboost import XGBClassifier

# seed
RANDOM_STATE_SEED = 42
# train-test-split test size
TEST_SIZE = 0.15
# logistic regression model constants
LOGISTIC_REGRESSION_ITER = 10000



def eval_model(model, X, y):
    predictions = model.predict(X)
    print(metrics.confusion_matrix(y, predictions))
    print(metrics.classification_report(y, predictions))

    
def runGridSearchWithDataset(estimator, hyper_params, X_train, X_test, y_train, y_test, sample_weight=None):
    # determine best parameters to fit model ti training data
    grid_search = GridSearchCV(
        estimator = estimator
        ,param_grid = hyper_params
        ,scoring = None
        ,n_jobs = 1
        ,cv = 5
        ,verbose = 0
        ,return_train_score = False
    )

    best_model = grid_search.fit(X_train, y_train, sample_weight=sample_weight)
    
    for k in hyper_params.keys():
        print('Best ' + k + ':', best_model.best_estimator_.get_params()[k])

    eval_model(best_model, X_test, y_test)
    eval_model(best_model, X_train, y_train)
    return best_model


def runGridSearch(estimator, hyper_params, labeled_df, X_cols, y_col):
    X_train, X_test, y_train, y_test = train_test_split(labeled_df[X_cols], labeled_df[y_col], test_size=TEST_SIZE, random_state=RANDOM_STATE_SEED)
    return runGridSearchWithDataset(estimator, hyper_params, X_train, X_test, y_train, y_test)
    
#     # determine best parameters to fit model ti training data
#     grid_search = GridSearchCV(
#         estimator = estimator
#         ,param_grid = hyper_params
#         ,scoring = None
#         ,n_jobs = 1
#         ,cv = 5
#         ,verbose = 0
#         ,return_train_score = False
#     )

#     best_model = grid_search.fit(X_train, y_train)
    
#     for k in hyper_params.keys():
#         print('Best ' + k + ':', best_model.best_estimator_.get_params()[k])

#     eval_model(best_model, X_test, y_test)
#     eval_model(best_model, X_train, y_train)
#     return best_model

def showLogRegResults(best_model_log, X_cols):
    log_coefs = pd.DataFrame(data=best_model_log.best_estimator_.coef_.T, columns=['Coef'], index=X_cols)
    log_coefs['Absolute Value'] = log_coefs.abs().mean(axis=1)
    log_coefs = log_coefs.reindex(log_coefs.abs().sort_values(by='Absolute Value', ascending=False).index)
    
    log_coefs_20_fig, log_coefs_20_axes = plt.subplots()

    temp_data = log_coefs.head(20)['Absolute Value']

    sns.barplot(y=temp_data.index, x=temp_data.values, ax=log_coefs_20_axes)

    # set text
    log_coefs_20_axes.set_title('LogReg: Top 20 Predictors of Diagnostic Error Level')
    log_coefs_20_axes.set_ylabel('Data Field')
    log_coefs_20_axes.set_xlabel('Relative Strength of Prediction')

    
def runLogReg(hyper_params, labeled_df, X_cols, y_col):
    estimator_log = LogisticRegression(max_iter=LOGISTIC_REGRESSION_ITER, random_state=RANDOM_STATE_SEED, class_weight='balanced')
    best_model_log = runGridSearch(estimator_log, hyper_params, labeled_df, X_cols, y_col)
    showLogRegResults(best_model_log, X_cols)
    return best_model_log


def runLogRegWithDataset(hyper_params, X_cols, X_train, X_test, y_train, y_test, sample_weight=None):
    estimator_log = LogisticRegression(max_iter=LOGISTIC_REGRESSION_ITER, random_state=RANDOM_STATE_SEED)
    #estimator_log = LogisticRegression(max_iter=LOGISTIC_REGRESSION_ITER, random_state=RANDOM_STATE_SEED, class_weight='balanced')
    best_model_log = runGridSearchWithDataset(estimator_log, hyper_params, X_train, X_test, y_train, y_test, sample_weight)
    showLogRegResults(best_model_log, X_cols)
    return best_model_log


def showRandForestResults(best_model_rf, X_cols):
    importances = pd.DataFrame(data=best_model_rf.best_estimator_.feature_importances_, columns=['Feature Importance'], index=X_cols)
    importances = importances.sort_values(by=['Feature Importance'], ascending=False)
    # create figure variables
    importances_20_fig, importances_20_axes = plt.subplots()

    temp_data = importances.head(20)['Feature Importance']
    sns.barplot(y=temp_data.index, x=temp_data.values, ax=importances_20_axes)

    # set text
    importances_20_axes.set_title('RF: Top 20 Predictors of Diagnostic Error Level')
    importances_20_axes.set_ylabel('Data Field')
    importances_20_axes.set_xlabel('Relative Strength of Prediction')


def runRandForest(hyper_params, labeled_df, X_cols, y_col):
    estimator_rf = RandomForestClassifier(random_state=RANDOM_STATE_SEED, class_weight='balanced')
    best_model_rf = runGridSearch(estimator_rf, hyper_params, labeled_df, X_cols, y_col)
    showRandForestResults(best_model_rf, X_cols)
    return best_model_rf


def runRandForestWithDataset(hyper_params, X_cols, X_train, X_test, y_train, y_test, sample_weight=None):
    estimator_rf = RandomForestClassifier(random_state=RANDOM_STATE_SEED, class_weight='balanced')
    best_model_rf = runGridSearchWithDataset(estimator_rf, hyper_params, X_train, X_test, y_train, y_test, sample_weight)
    showRandForestResults(best_model_rf, X_cols)
    return best_model_rf
