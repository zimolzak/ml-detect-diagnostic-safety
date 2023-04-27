.PHONY: all clean
files = pert.pdf wordvec-similarity.pdf vinci_technical.pdf handoff_doc.pdf trigger-flow.png prior-abstract.docx zimolzak-sidm-abstract.docx

all: $(files)

%.pdf: %.tex
# technically wordvec-similarity only needs pdflatex once
	pdflatex $<
	pdflatex $<

%.docx: %.txt
	pandoc -o $@ $<

pert.pdf: pert.dot
	dot -Tpdf -o $@ $<

trigger-flow.png: trigger-flow.dot
	dot -Tpng -o $@ $<

clean:
	rm -f $(files)
	rm -f *.aux *.log *.out *.toc
