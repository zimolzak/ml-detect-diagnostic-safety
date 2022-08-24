.PHONY: all clean
files = pert.pdf wordvec-similarity.pdf vinci_technical.pdf

all: $(files)

vinci_technical.pdf: vinci_technical.tex
	pdflatex $<
	pdflatex $<

wordvec-similarity.pdf: wordvec-similarity.tex
	pdflatex $<

%.pdf: %.dot
	dot -Tpdf -o $@ $<

clean:
	rm -f $(files)
	rm -f *.aux *.log *.out *.toc
