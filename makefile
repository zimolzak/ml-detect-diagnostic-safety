.PHONY: all clean

csvoutputs = tidy-icd10-dizzy.csv tidy-icd10-headache.csv		\
tidy-icd10-stroke-risk.csv tidy-icd10-stroke.csv tidy-icd9-dizzy.csv	\
tidy-icd9-headache.csv tidy-icd9-stroke-risk.csv tidy-icd9-stroke.csv

all: $(csvoutputs)
	$(MAKE) -C AndyZimolzak

tidy-%.csv: %.csv build_csvs.py
	python build_csvs.py

clean:
	rm -f $(csvoutputs)
	$(MAKE) -C AndyZimolzak clean
