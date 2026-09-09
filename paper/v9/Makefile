# Build this snapshot without modifying paper/main.* or any earlier version.
PYTHON ?= python3
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
TEXSRC := main.tex assets/macros.tex assets/references.bib $(wildcard sections/*.tex)
FIGS := $(wildcard figures/*.pdf)

.PHONY: all verify figures
all: main.pdf

main.pdf: $(TEXSRC) $(FIGS)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIBTEX) main
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex

verify:
	$(PYTHON) verify_snapshot.py

# Requires matplotlib and numpy. Render from the committed results in this repo.
figures:
	$(PYTHON) make_review_figures.py
	$(PYTHON) make_secondary_figures.py
