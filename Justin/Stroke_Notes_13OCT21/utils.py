import os
import pathlib
import json
from tqdm import tqdm
from collections import Counter

def extract_umls_cuis(save = False):
    cuimonolith = dict()
    for i in tqdm(pathlib.Path("ClampOutput/").glob("*.txt"), total=30087): #hard coded total
        with open(i,'r') as infile:
            for line in infile:
                line = line.strip().split('\t')
                if line[0] != 'NamedEntity':
                    break
                semtype, presence, cui = line[3:6]
                if i.stem not in cuimonolith.keys():
                    cuimonolith[i.stem] = {'cuis':[], 'semtype':[], 'presence':[]}
                cuimonolith[i.stem]['cuis'] += [cui.split("=")[1]]
                cuimonolith[i.stem]['semtype'] += [semtype.split("=")[1]]
                cuimonolith[i.stem]['presence'] += [presence.split("=")[1]]
    if save:
        with open("sids_to_clamp_cuis.json",'w') as outfile:
            json.dump(cuimonolith, outfile)
    return cuimonolith


def generate_cui_translation_dictionary(save = False):
    cuitranslate = dict()
    with open("MRCONSO.RRF",'r', encoding='UTF-8') as infile:
        for line in infile:
            line = line.strip().split('|')
            engpy = line[1]+line[2]+line[4]+line[6]
            if engpy == "ENGPPFY":
                cuitranslate[line[0]] = line[14]
    if save:
        with open("cuitranslate.json",'w') as outfile:
            json.dump(cuitranslate, outfile)
    return cuitranslate