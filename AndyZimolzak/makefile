.PHONY: all clean
files = pert.pdf wordvec-similarity.pdf

all: $(files)

wordvec-similarity.pdf: wordvec-similarity.tex
	pdflatex $<

%.pdf: %.dot
	dot -Tpdf -o $@ $<

clean:
	rm -f $(files)
	rm -f *.aux *.log
