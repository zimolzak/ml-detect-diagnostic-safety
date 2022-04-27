.PHONY: all
files = pert.pdf wordvec-similarity.pdf
csvoutputs = tidy-icd10-dizzy.csv tidy-icd10-headache.csv tidy-icd10-stroke-risk.csv tidy-icd10-stroke.csv tidy-icd9-dizzy.csv tidy-icd9-headache.csv tidy-icd9-stroke-risk.csv tidy-icd9-stroke.csv

all: $(files) $(csvoutputs)

tidy-%.csv: build_csvs.py icd10-dizzy.csv icd10-headache.csv icd10-stroke-risk.csv icd10-stroke.csv icd9-dizzy.csv icd9-headache.csv icd9-stroke-risk.csv icd9-stroke.csv
	python build_csvs.py

wordvec-similarity.pdf: wordvec-similarity.tex
	pdflatex $<

%.pdf: %.dot
	dot -Tpdf -o $@ $<

clean:
	rm -f $(files) $(csvoutputs)
