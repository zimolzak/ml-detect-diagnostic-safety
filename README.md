# Machine Learning to Enhance Detection of Diagnostic Safety Events

Contents overview current as of 2022-04-28, commit id b066d83.

- root
    - ICD code groupers (dizzy, headache, stroke-risk, stroke)
    - medication groupers
- AIM2/
    - "Original" classifier ipynb? Applies sklearn to SPADE data.
    - Some old scripts to generate fake data.
    - Mostly from late 2020.
- AndyZimolzak/
    - ipynb, SQL, and RPT files serving as an intro to structured data
    - SQL and TXT files related to note titles (AMA, discharge, and in general)
    - slightly modified classifier.ipynb
    - TXT file listing conda packages
    - python helper functions and tests
    - Graphviz file of PERT diagram of onboarding
    - ipynb to inspect a few UMAP'd discharge summaries from Max
    - LaTeX notation about discharge summary embeddings
- Justin/
    - The things we use the most are in the Stroke_Notes_13OCT21 folder.
    - cui semi-manual filtering ipynb
    - word2vec ipynb (pytorch)
    - UMAP "octopus" 1Hot ipynb
    - UMAP "whale" CUIDocumentVectors.ipynb
    - CSV of discharge summ note titles, adjudicated by Andy
    - TXTs of 140 note types, 90 disch note titles
- Li/
    - ipynb about comorbidity index
    - SQL for SPADE data?
- Maxxxx/
    - ipynb continuing Justin's UMAP of CUIs. Outlier disch summaries.
    - UMAP of structured "B00" data
- Paarth/
    - ipynb to input CSV, output SQL
    - SQL with ICD groupers, SPADE-related



Note: the SPADE comorbidity index is calculated using [SQL code from a different
repo](https://github.com/paarth-kapadia/saferdx_hou/tree/main/z_util/comorbidity_index_elix).
Two files named `proto_sf_calc_elix.sql` and `table_sf_calc_elix.sql`.



## Bundle

`git bundle create your-filename.bundle HEAD main`
