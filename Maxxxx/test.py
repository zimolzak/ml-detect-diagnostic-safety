import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from utils import extract_umls_cuis, generate_cui_translation_dictionary
import json
import umap
import scipy
from collections import Counter
import matplotlib

with open('..\Justin\Stroke_Notes_13OCT21\superposed_document_vectors.json','r') as infile:
    docvectors = json.load(infile)

docids = list(docvectors.keys())
docarray = np.asarray([docvectors[x] for x in docids])

reducer = umap.UMAP(n_components=3,init='random',random_state=0)

embedding = reducer.fit_transform(docarray)

import matplotlib.pyplot as plt
plt.style.use("dark_background")

fig = plt.figure(figsize=(12,8), dpi=100)
ax = fig.add_subplot(projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.3)
plt.show()