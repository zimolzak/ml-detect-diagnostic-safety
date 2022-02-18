.PHONY: all
files = pert.pdf

all: $(files)

%.pdf: %.dot
	dot -Tpdf -o $@ $<

clean:
	rm -f $(files)
