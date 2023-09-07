.PHONY: all clean
files = wordvec-similarity.pdf vinci_technical.pdf handoff_doc.pdf  # LaTeX
files += pert.pdf trigger-flow.png  # Graphviz
files += prior-abstract.docx zimolzak-sidm-abstract.docx  # pandoc
files += flow-diagram.pdf flow-diagram.png with-subgraph.pdf  # Graphviz for talk

all: $(files)

# LaTeX

%.pdf: %.tex
        # technically wordvec-similarity only needs pdflatex once
	pdflatex $<
	pdflatex $<

# Pandoc

%.docx: %.txt
	pandoc -o $@ $<

# Graphviz

pert.pdf: pert.dot
	dot -Tpdf -o $@ $<

flow-diagram.pdf: flow-diagram.dot
	dot -Tpdf -o $@ $<

with-subgraph.pdf: with-subgraph.dot
	dot -Tpdf -o $@ $<

%.png: %.dot
	dot -Tpng -o $@ $<

# Clean

clean:
	rm -f $(files)
	rm -f *.aux *.log *.out *.toc
