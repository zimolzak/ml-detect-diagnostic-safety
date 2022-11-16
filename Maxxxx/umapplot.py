import numpy as np
# import pandas as pd
# from glob import glob
# from utils import extract_umls_cuis, generate_cui_translation_dictionary
import json
import umap
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from collections import Counter


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


def standard_docvec_builder(cuidata, cuiarray, vocab, normalize):
    doc_vectors = np.zeros((len(cuidata), cuiarray.shape[1]))
    for i,doc in tqdm(enumerate(cuidata), total=len(cuidata)):
        if len(cuidata[doc]['cuis']) == 0:
            print("doc has no cuis:", doc)
            continue
        for cui in cuidata[doc]['cuis']:
            doc_vectors[i] += cuiarray[vocab[cui]]
        if normalize:
            doc_vectors[i] /= len(cuidata[doc]['cuis'])
    return doc_vectors


def neg_absent_docvec_builder(cuidata, cuiarray, vocab, normalize):
    missing_vec = Counter()
    doc_vectors = np.zeros((len(cuidata), cuiarray.shape[1]))
    for i,doc in tqdm(enumerate(cuidata), total=len(cuidata)):
        if len(cuidata[doc]['cuis']) == 0:
            print("doc has no cuis:", doc)
            continue
        for cui, presence in zip(cuidata[doc]['cuis'], cuidata[doc]['presence']):
            if cui not in vocab:
                missing_vec.update([cui])
                continue
            if presence == 'present':
                doc_vectors[i] += cuiarray[vocab[cui]]
            else:
                doc_vectors[i] -= cuiarray[vocab[cui]]
        if normalize:
            doc_vectors[i] /= len(cuidata[doc]['cuis'])
    print(len(missing_vec))
    #     print(missing_vec)
    return doc_vectors


def presence_dominant_docvec_builder(cuidata, cuiarray, vocab, normalize):
    with open("dominant_cui_presence.json",'r') as infile:
        dominant_presences = json.load(infile)
    doc_vectors = np.zeros((len(cuidata), cuiarray.shape[1]))
    for i,doc in tqdm(enumerate(cuidata), total=len(cuidata)):
        if len(cuidata[doc]['cuis']) == 0:
            print("doc has no cuis:", doc)
            continue
        for cui, presence in zip(cuidata[doc]['cuis'], cuidata[doc]['presence']):
            if cui not in vocab:
                print("cui missing from vectors:", cui)
                continue
            # if presence is the form we trained on, +
            if dominant_presences[cui] == (presence == "present"):
                doc_vectors[i] += cuiarray[vocab[cui]]
            else:
                doc_vectors[i] -= cuiarray[vocab[cui]]
        if normalize:
            doc_vectors[i] /= len(cuidata[doc]['cuis'])
    #     print(missing_vec)
    return doc_vectors

def do_everything(cui_vec_filename, sid2cui_filename, reducer, mapper=standard_docvec_builder, normalize=True):
    with open(cui_vec_filename,'r') as infile:
        cuivectors = json.load(infile)
    # load cui vectors
    cuiids = list(cuivectors.keys())
    cuiarray = np.asarray([cuivectors[x] for x in cuiids])
    vocab = {cuiids[i]: i for i in range(len(cuiids))}
    print(cuiarray.shape)
    
    with open(sid2cui_filename,'r') as infile:
        cuidata = json.load(infile)
    
    # build doc vectors from cui vectors
    doc_vectors = mapper(cuidata, cuiarray, vocab, normalize)
    
    embedding = reducer.fit_transform(doc_vectors)
    
    plt.style.use("dark_background")
    
    fig = plt.figure(figsize=(12,8), dpi=100)
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.3)
    plt.show()
    
    return cuiids, cuiarray, vocab, cuidata, doc_vectors, embedding


def highlight_icds(docids, embedding, docidstoicds, highlight_icds, title):
    icdcodefordocid = dict()
    for docid,icds in docidstoicds.items():
        code = False
        for icd in icds:
            if icd in highlight_icds:
                code = True
                break
        if code:
            icdcodefordocid[docid] = 1
        else:
            icdcodefordocid[docid] = 0
    icdcodebyindex = [icdcodefordocid[docid] for docid in docids]
    fig = plt.figure(figsize=(12,8), dpi=100)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=icdcodebyindex, alpha=0.3, cmap='Set1')
    plt.title(title)
    plt.show()

def plot_highlights(cuidata, embeddings):
    with open("docidtooldesticds.json", 'r') as infile:
        docidstoicds = json.load(infile)
    
    docids = cuidata.keys()
    
    # plot stroke icds
    strokeicd10s = set(['G43.609', 'G43.619', 'G43.601', 'G43.611', 'I60.00', 'I60.01', 'I60.02', 'I60.10', 'I60.11', 'I60.12', 'I60.2', 
                    'I60.30', 'I60.31', 'I60.32', 'I60.4', 'I60.50', 'I60.51', 'I60.52', 'I60.6', 'I60.7', 'I60.8', 'I60.9', 'I61.0', 
                    'I61.1', 'I61.2', 'I61.3', 'I61.4', 'I61.5', 'I61.6', 'I61.8', 'I61.9', 'I62.1', 'I62.00', 'I62.01', 'I62.02', 
                    'I62.03', 'I62.9', 'I63.02', 'I63.12', 'I63.22', 'I63.031', 'I63.032', 'I63.033', 'I63.039', 'I63.131', 'I63.132', 
                    'I63.133', 'I63.139', 'I63.231', 'I63.232', 'I63.233', 'I63.239', 'I63.011', 'I63.012', 'I63.013', 'I63.019', 'I63.111', 
                    'I63.112', 'I63.113', 'I63.119', 'I63.211', 'I63.212', 'I63.213', 'I63.219', 'I63.59', 'I63.09', 'I63.19', 'I63.59',
                    'I63.00', 'I63.10', 'I63.20', 'I63.29', 'I66.01', 'I66.02', 'I66.03', 'I66.09', 'I66.11', 'I66.12', 'I66.13', 'I66.19', 
                    'I66.21', 'I66.22', 'I66.23', 'I66.29', 'I66.3', 'I66.8', 'I66.9', 'I63.30', 'I63.311', 'I63.312', 'I63.313', 'I63.319', 
                    'I63.321', 'I63.322', 'I63.323', 'I63.329', 'I63.331', 'I63.332', 'I63.333', 'I63.339', 'I63.341', 'I63.342', 'I63.343', 
                    'I63.349', 'I63.39', 'I63.6', 'I63.40', 'I63.411', 'I63.412', 'I63.413', 'I63.419', 'I63.421', 'I63.422', 'I63.423',
                    'I63.429', 'I63.431', 'I63.432', 'I63.433', 'I63.439', 'I63.441', 'I63.442', 'I63.443', 'I63.449', 'I63.49', 'I63.50', 
                    'I63.511', 'I63.512', 'I63.513', 'I63.519', 'I63.521', 'I63.522', 'I63.523', 'I63.529', 'I63.531', 'I63.532', 'I63.533', 
                    'I63.539', 'I63.541', 'I63.542', 'I63.543', 'I63.549', 'I63.59', 'I63.8', 'I63.81', 'I63.89', 'I63.9', 'I67.89', 'I65.1', 
                    'I65.21', 'I65.22', 'I65.23', 'I65.29', 'I65.01', 'I65.02', 'I65.03', 'I65.09', 'I65.8', 'I65.9', 'I67.2', 'I67.81', 
                    'I67.82', 'I67.89', 'I67.1', 'I67.7', 'I68.2', 'I67.5', 'I67.6', 'G45.4', 'G46.3', 'G46.4', 'G46.5', 'G46.6', 'G46.7', 
                    'G46.8', 'I67.89', 'I68.0', 'I68.8', 'I67.9', 'G45.0', 'G45.8', 'G45.1', 'G45.2', 'G45.8', 'G46.0', 'G46.1', 'G46.2', 
                    'G45.9', 'I67.841', 'I67.848'])
    highlight_icds(docids, embeddings, docidstoicds, strokeicd10s, "strokeicd10s")
    
    # plot dizziness icds
    dizzinessicd10s = ['H81.01', 'H81.02', 'H81.03', 'H81.09', 'H81.10', 'H81.11', 'H81.12', 'H81.13', 'H81.20', 'H81.21', 'H81.22', 'H81.23', 'H81.311', 
                   'H81.312', 'H81.313', 'H81.319', 'H81.391', 'H81.392', 'H81.393', 'H81.399', 'H81.4', 'H81.41', 'H81.42', 'H81.43', 'H81.49', 
                   'H81.8X1', 'H81.8X2', 'H81.8X3', 'H81.8X9', 'H81.90', 'H81.91', 'H81.92', 'H81.93', 'H83.01', 'H83.02', 'H83.03', 'H83.09', 'H83.11', 
                   'H83.12', 'H83.13', 'H83.19', 'H83.2X1', 'H83.2X2', 'H83.2X3', 'H83.2X9', 'H83.3X1', 'H83.3X2', 'H83.3X3', 'H83.3X9', 'H83.8X1', 
                   'H83.8X2', 'H83.8X3', 'H83.8X9', 'H83.90', 'H83.91', 'H83.92', 'H83.93', 'R42.']
    highlight_icds(docids, embeddings, docidstoicds, dizzinessicd10s, "dizzinessicd10s")


    # plot headaches
    headacheicd10s = ['G43.001', 'G43.009', 'G43.011', 'G43.019', 'G43.101', 'G43.109', 'G43.111', 'G43.119', 'G43.401', 'G43.409', 'G43.411', 'G43.419', 
                  'G43.501', 'G43.509', 'G43.511', 'G43.519', 'G43.601', 'G43.609', 'G43.611', 'G43.619', 'G43.701', 'G43.709', 'G43.711', 'G43.719', 
                  'G43.801', 'G43.809', 'G43.811', 'G43.819', 'G43.821', 'G43.829', 'G43.831', 'G43.839', 'G43.901', 'G43.909', 'G43.911', 'G43.919', 
                  'G43.A0', 'G43.A1', 'G43.B0', 'G43.B1', 'G43.C0', 'G43.C1', 'G43.D0', 'G43.D1', 'G44.001', 'G44.009', 'G44.011', 'G44.019', 'G44.021', 
                  'G44.029', 'G44.031', 'G44.039', 'G44.041', 'G44.049', 'G44.051', 'G44.059', 'G44.091', 'G44.099', 'G44.1', 'G44.201', 'G44.209', 
                  'G44.211', 'G44.219', 'G44.221', 'G44.229', 'G44.301', 'G44.309', 'G44.311', 'G44.319', 'G44.321', 'G44.329', 'G44.40', 'G44.41', 
                  'G44.51', 'G44.52', 'G44.53', 'G44.59', 'G44.81', 'G44.82', 'G44.83', 'G44.84', 'G44.85', 'G44.89', 'R51.']
    highlight_icds(docids, embeddings, docidstoicds, headacheicd10s, "headacheicd10s")
    