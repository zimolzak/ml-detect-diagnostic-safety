import numpy as np
# import pandas as pd
# from glob import glob
# from utils import extract_umls_cuis, generate_cui_translation_dictionary
import json
import umap
import matplotlib
import matplotlib.pyplot as plt


def reduce_and_plot(filename, reducer):
    with open(filename,'r') as infile:
        docvectors = json.load(infile)

    docids = list(docvectors.keys())
    docarray = np.asarray([docvectors[x] for x in docids])
    
    embedding = reducer.fit_transform(docarray)
    
    plt.style.use("dark_background")
    
    fig = plt.figure(figsize=(12,8), dpi=100)
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.3)
    plt.show()
    
    return docids, docarray