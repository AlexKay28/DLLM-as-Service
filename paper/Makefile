# Build the paper. See BUILD.md for how to get the toolchain onto a fresh VM.
#
#   make            figures + paper + publish
#   make figures    rebuild all 9 figures from results/*.json
#   make paper      compile main.tex -> main.pdf
#   make verify     assert every cited number against its artifact
#   make publish    snapshot main.pdf into paper/$(VERSION)/main.pdf
#   make check      report the toolchain versions and the resolved figure font
#   make clean      remove LaTeX build droppings (keeps main.pdf)
#   make distclean  also remove main.pdf and the generated figures

SHELL     := /bin/bash
PAPER     := main
REPO      := $(abspath ..)
# Released snapshots live in paper/<VERSION>/main.pdf — bump VERSION to cut a
# new one rather than overwriting the last.
VERSION   := v7
RELEASE   := $(VERSION)/main.pdf

# Override if your interpreter is named differently: make PYTHON=python3.12
PYTHON    ?= python3
TECTONIC  ?= tectonic

FIGSRC    := figures/make_figures.py figures/style.py
FIGS      := $(patsubst %,figures/%.pdf, \
               fig1_cost_shapes fig2_canvas_edges fig3_input_sweep fig4_ksweep \
               fig5_quality_vs_length fig6_batching \
               fig8_defects fig9_deployment \
               fig10_usecase_chains fig11_usecase_schema)
TEXSRC    := $(PAPER).tex assets/style.tex assets/macros.tex assets/references.bib $(wildcard sections/*.tex)

.PHONY: all figures paper verify publish check clean distclean html

all: publish

# ---------------------------------------------------------------- figures --
# Every figure is rebuilt from results/*.json in one pass, so the whole set is
# regenerated whenever either source file changes.
figures: $(FIGS)

$(FIGS): $(FIGSRC)
	@echo ">> building figures"
	@cd figures && $(PYTHON) make_figures.py

# ------------------------------------------------------------------ paper --
paper: $(PAPER).pdf

$(PAPER).pdf: $(TEXSRC) $(FIGS)
	@echo ">> compiling $(PAPER).tex"
	@pdflatex -interaction=nonstopmode $(PAPER).tex
	@bibtex $(PAPER)
	@pdflatex -interaction=nonstopmode $(PAPER).tex
	@pdflatex -interaction=nonstopmode $(PAPER).tex
	@$(PYTHON) -c "import re; log=open('$(PAPER).log').read(); print('>> %d overfull box(es), %d undefined reference(s)' % (len(re.findall(r'Overfull .hbox', log)), len(re.findall(r'Reference .* undefined', log))))"

# ----------------------------------------------------------------- verify --
# Re-reads every artifact cited in NUMBERS.md and CHECKLIST.md. Needs 3.10+
# (provenance.py uses PEP 604 unions).
verify:
	@echo ">> verifying cited numbers against results/"
	@cd $(REPO) && $(PYTHON) paper/verify_numbers.py

# ---------------------------------------------------------------- publish --
publish: $(PAPER).pdf
	@mkdir -p $(VERSION)
	@cp $(PAPER).pdf $(RELEASE)
	@echo ">> published paper/$(RELEASE) ($$(pdfinfo $(RELEASE) 2>/dev/null | awk '/Pages/{print $$2}') pages)"

# ------------------------------------------------------------------ check --
check:
	@echo "python:    $$($(PYTHON) --version 2>&1)"
	@echo "tectonic:  $$($(TECTONIC) --version 2>&1)"
	@$(PYTHON) -c "import matplotlib, numpy; print('matplotlib:', matplotlib.__version__); print('numpy:     ', numpy.__version__)"
	@cd figures && $(PYTHON) -c "import style; style.apply(); print('figure font:', style.resolved_font())"

# ------------------------------------------------------------------ clean --
clean:
	@rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).bbl $(PAPER).blg \
	       $(PAPER).fls $(PAPER).fdb_latexmk $(PAPER).synctex.gz
	@echo ">> cleaned build files"

distclean: clean
	@rm -f $(PAPER).pdf figures/fig*.pdf figures/fig*.png
	@echo ">> removed main.pdf and generated figures"

# ------------------------------------------------------------------- html --
# Generate HTML with MathJax. Fix FiraSans ligature rendering (ff -> 0).
html: $(TEXSRC) $(FIGS)
	@echo ">> building HTML"
	@htlatex $(PAPER).tex "xhtml,mathjax,charset=utf-8" 2>&1 | tail -3
	@sed -i 's/Di0usion/Diffusion/g; s/di0usion/diffusion/g' $(PAPER).html
	@echo ">> $(PAPER).html ($(shell wc -l < $(PAPER).html) lines)"
