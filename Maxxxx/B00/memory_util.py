import os
import gc
import string
import numpy as np
import pandas as pd
import pyodbc

# Default free space allocation size. All memory operations in this class will keep at least this amount free. Change if needed
FREE_SPACE = 2048


def get_memory_status():
    process = os.popen('systeminfo | find "Virtual Memory"')
    result = process.read()
    process.close()
    
    d = dict()
    # the following logic will only make sense once printed
    for s in result.splitlines():
        print(s)
        tokens = s.split(": ")
        
        name = tokens[1].replace(" ","")
        
        size_tokens = list(filter(None, tokens[2].split(" ")))
        
        assert(size_tokens[1] == "MB") # assert the unit is still MB
        
        d[name] = int(size_tokens[-2].replace(",",""))
    
    return d


class LazyCache:
    def __init__(self):
        self.values = dict()
        self.lambdas = dict()
        
    def __getitem__(self, key):
        if key in self.values:
            return self.values[key]
        
        if key not in self.lambdas:
            raise Exception("Need to configure a generating lambda first")
        
        mem_status = get_memory_status()
        while mem_status["Available"] < FREE_SPACE:
            self.pop_cache()
        
        v = self.lambdas[key]()
        self.values[key] = v
        return v
    
    def keys(self):
        return self.lambdas.keys()
        
    def configure_lambda(self, key, f):
        self.lambdas[key] = f
        
    def pop_cache(self):
        # TODO: there are much more efficient/effective ways to do cache eviction
        if not self.values:
            raise Exception("Cache already empty, no more to pop")
        
        max_key, max_size = None, -1
        for k, v in self.values.items():
            s = np.prod(v.shape) # here it's assumed to be numpy or pandas array
            if s > max_size:
                max_key = k
                max_size = s
                
        del self.values[max_key]
        gc.collect()
        
    def clear(self):
        self.values.clear()
        gc.collect()

"""
If necessary, one can also implement a eviction strategy that measures and keeps track of the ACTUAL memory usage of the values.
This can be done using:

import tracemalloc

tracemalloc.start()

v = self.lambdas[key]()

tracemalloc.get_traced_memory()[0] / (1024 ** 2) # convert to MB
"""
        
    
    
    
# Cache variation of feature_util's extractDataset
# lazily extracts all datasets from database with selected prefix
def extractDataset(prefix, excludeSet):
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=VHACDWRB03;DATABASE=ORD_Singh_201911038D")

    info_df = pd.read_sql(sql="select * from information_schema.tables where table_name like '%{}%'".format(prefix), con=conn)
    display(info_df)
    # read all the tables into pandas tables
    tables = LazyCache()
    for tname in info_df.TABLE_NAME:
        if tname.split('_')[-1] in excludeSet:
            continue
        query_str = "select * from  " + str("Dflt.")+ str(tname)
        tables.configure_lambda(tname.split('_')[-1], lambda: pd.read_sql(sql=query_str,con=conn))
    print(tables.keys())
    return tables