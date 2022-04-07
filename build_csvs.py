import pandas as pd

LS = """icd10-dizzy.csv
icd10-headache.csv
icd10-stroke-risk.csv
icd10-stroke.csv
icd9-dizzy.csv
icd9-headache.csv
icd9-stroke-risk.csv
icd9-stroke.csv
"""

FILENAMES = LS.split()

if __name__ == '__main__':
    for name in FILENAMES:
        df = pd.read_csv(name)
        prefix = name.split('.')[0]
        new_name = prefix + '-tidy.csv'
        df.to_csv(new_name)
