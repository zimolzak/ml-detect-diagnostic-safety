# IMPORTS
# Data Libraries
import pandas as pd
import numpy as np

# Util Libraries
import random
from datetime import datetime, timedelta
import os

# set working path 
WORKING_PATH = "./"

# set input directory
INPUT_DIRECTORY_FILEPATH = os.path.join(WORKING_PATH, 'Input')

# column metadata filepath
COLUMN_METADATA_FILEPATH = os.path.join(INPUT_DIRECTORY_FILEPATH, 'column_metadata.csv')

# this controls number of rows of fake data generated
NROWS = 200



# DATA GENERATION
# helper function to generate a random datetime
def generateRandomDateTime(yearFloor=2000, yearCeil=datetime.now().year):
    start = datetime(yearFloor, 1, 1, 00, 00, 00)
    years = yearCeil - yearFloor + 1
    end = start + timedelta(days=365 * years)
    
    return start + (end - start) * random.random()


# return a column of data of type dataType drawing values from dataRange
def generateData(dataType, dataRange):
    data = []
    if(dataType == 'id'):
        data = list(range(NROWS))
    elif(dataType == 'datetime'):
        data = [generateRandomDateTime() for _ in range(NROWS)]
    elif(dataType == 'intcat'):
        data = [int(dataRange[random.randint(0, len(dataRange) - 1)]) for _ in range(NROWS)]
    elif(dataType == 'stringcat'):
        data = [dataRange[random.randint(0, len(dataRange) - 1)] for _ in range(NROWS)]
    elif(dataType == 'intcon'):
        data = [random.randint(int(dataRange[0]), int(dataRange[1])) for _ in range(NROWS)]
    elif(dataType == 'floatcon'):
        data = [random.uniform(float(dataRange[0]), float(dataRange[1])) for _ in range(NROWS)]
    elif(dataType == 'stringu'):
        data = ['random text' for _ in range(NROWS)]
    else:
        print(dataType)
        print('somethings wrong')
        raise

    return data



# IO
# read data descriptions
def read_data_description():
    data_description = pd.read_csv(COLUMN_METADATA_FILEPATH)[0:-1]
    data_description['Data Range'] = data_description['Data Range'].apply(lambda x:  x.split(' | ') if type(x) is str and ' | ' in x else x)
    
    return data_description


# generate data frame
def make_dataframe():
    data_description = read_data_description()
    cols = []
    for i in range(data_description['Name'].shape[0]):
        col = generateData(data_description.iloc[i]['Data Type'], data_description.iloc[i]['Data Range'])
        cols.append(col)
    df = pd.DataFrame(cols).transpose()
    df.columns = data_description['Name'].values
    
    return df
