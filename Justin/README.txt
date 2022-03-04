This directory contains 2 project folders of note:

1. StrokeReRanking
	a. This work is a continuation of Paarth's excellent work.
	b. Description:
		- Training Data Description: data is derived from ~300 patient records that were trigger positive for the Gen 1 dizziness / stroke / SPADE trigger
		- A simple model was trained using linear models, and can be found in the subdirectory `paarthwork`
		- Some experiments using note information was undertaken and is in subdirectory `note_experiments` (these are archived, so any links in notebooks may need to be updated for them to run)
		- 7zip exists here if you need to decompress .7z files (note that the standard workspace can decompress 7zip files; only the VINCI desktop lacks this ability, so often you can use the standard desktop to decompress, then jump over to the vinci space)
		- The paarth subdirectory:
			- This directory contains the bulk of the important experiments in various ipynb notebooks, most carried over or enhanced from paarths work. 
			- This directory was originally a clone of Paarth's.
			- Code here is not necessarily optimized, and was written sometimes by me, sometimes by paarth.
		- invivotestspade were rankings of a new data pull to assess the possibility of ML re-ranking of trigger positive records (e.g. if a reviewer could only look at 10 records, would ML be able to give them 10 probably MOD better than taking 10 at random from trigger positives?). These were meant to be analyzed by a clinician to assess the success of this endeavour.
2. Stroke_Notes_13OCT21
	a. This work is primarily focused on taking a broad view of stroke, and attempting to build an embedding space for it. The idea was, if we can build a good embedding for discharge summaries / for patients in an encounter (a loaded term - Andy / others can elucidate how an encounter might be described from a clinical perspective), then we could use that embedding space to find phenotypic subgroups of the stroke space (e.g. perhaps all posterior circulation strokes will be in one cluster, or perhaps all patients without CT / MRI / imaging will be in another cluster). This work ideally would utilize structured data (which is very information rich) and free text data (which has lots of signal but is extremely difficult to use computationally for a variety of reasons - review papers exist on this).
	b. As the name might suggest, we only had the notes to start, and specifically this focused on using CLAMP to extract UMLS CUIs (which could be one possible way to extract useful information from raw clinical text) from discharge summaries (TIUSTandardTitles as adjudicated by Andy).
	c. There exists an MRCONSO.RRF file which contains CUI to human readable mappings.
	d. utils.py contains python functions to extract CUIs from the CLAMP output and a function to translate CUIs to human readable forms.
	e. Various notebooks should be fairly self explanatory in title and purpose, with notes in the notebooks themselves.
	f. Initially, a UMAP of 1 hot coded (1 if a CUI existed in a document, 0 otherwise) was done to see if any structure might exist. 
	g. Results indicate some possible structure, but expectedly difficult to separate (structured data would likely result in much better clustering)
	h. A word2vec model was the next step to try and get better use out of the CUIs, which is the word2vec.ipynb.
	i. DCArchive is just various archived files (likely no need to sift through these).
	j. Note that the stroke_discharge_notes_adjudicated, which generated the ClampInput, is a malformed file and may have resulted in some malformed input for CLAMP (mostly due to incomplete quoting from SSMS output; ideally you'll want to connect to the sql table directly, and generate a fully quoted / delimited flat text file in python to generate the ClampInput files - this would require pyodbc and buy-in from VINCI IT to make sure you can actually access the SQL server tables [pyodbc is in the environment files as a driver, as well as sqlalchemy if access to the SQL tables via the devbox can be negotiated]).
	
Other files of note: there is an offline command line version of clamp, the cachefiles necessary for running some embedding experiments in strokereranking note experiments, this readme, and a tonsofstrokenotesdump file that Li generated which is very similar to the notes in Stroke_Notes_13OCT21 but contains all note types for patients. These aren't currently used in any analysis, but may prove useful at a later date (especially if direct SQL access via python isn't maintainable, though I'd recommend probably re-indexing these into a database to make querying much quicker than looking through non-indexed flat files).

# Notes by Andy 2021-11-24

File you may need to set up Anaconda packages is:

P:\ORD_Singh_201911038D\Upload\ahrq-probably-pytorch-2021-11.zip

Generally *do not* need to install each individually.
