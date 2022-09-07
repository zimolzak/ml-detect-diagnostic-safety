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


def split_cui_present_absent(save = False):
    with open("sids_to_clamp_cuis.json",'r') as infile:
        cuidata = json.load(infile)
    present_data = dict()
    absent_data = dict()
    for k,v in cuidata.items():
        p_cui = []
        p_semtype = []
        a_cui = []
        a_semtype = []
        for i in range(len(v['presence'])):
            if v['presence'][i] == "present":
                p_cui.append(v['cuis'][i])
                p_semtype.append(v['semtype'][i])
            else:
                a_cui.append(v['cuis'][i])
                a_semtype.append(v['semtype'][i])
        present_data[k] = {'cuis': p_cui, 'semtype':p_semtype}
        absent_data[k] = {'cuis': a_cui, 'semtype':a_semtype}
    if save:
        with open("sids_to_present_cuis.json",'w') as outfile:
            json.dump(present_data, outfile)
        with open("sids_to_absent_cuis.json",'w') as outfile:
            json.dump(absent_data, outfile)
    return present_data, absent_data
    
