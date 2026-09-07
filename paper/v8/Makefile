# Build this snapshot without modifying paper/main.* or any earlier version.
PYTHON ?= python3
TECTONIC ?= tectonic
TEXSRC := main.tex assets/macros.tex assets/references.bib $(wildcard sections/*.tex)
FIGS := $(wildcard figures/*.pdf)

.PHONY: all verify figures
all: main.pdf

main.pdf: $(TEXSRC) $(FIGS)
	$(TECTONIC) --keep-logs main.tex

verify:
	$(PYTHON) verify_snapshot.py

# Requires matplotlib and numpy. Render from the committed results in this repo.
figures:
	$(PYTHON) make_review_figures.py
	$(PYTHON) make_secondary_figures.py
