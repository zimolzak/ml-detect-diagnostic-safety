.PHONY: all clean

files = wordvec-similarity.pdf vinci_technical.pdf handoff_doc.pdf  # LaTeX
files += pert.pdf  # Graphviz
files += prior-abstract.docx zimolzak-sidm-abstract.docx  # pandoc

# Graphviz for talk

files += flow-colors-sparser.png flow-colors-sparser.pdf flow-colors-sparserLR.png

# Paper

files += flow-colors-sparser.eps




# Main

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

flow-colors-sparser.pdf: flow-colors-sparser.dot
	dot -Tpdf -o $@ $<

%.eps: %.dot  # New one for paper
	dot -Teps -o $@ $<

%.png: %.dot
	dot -Tpng -o $@ $<

%LR.png: %.dot
	dot -Tpng -Grankdir=LR -o $@ $<

# Clean

clean:
	rm -f $(files)
	rm -f *.aux *.log *.out *.toc
