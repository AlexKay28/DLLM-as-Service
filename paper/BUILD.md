# Building the paper on a fresh machine

Everything the paper needs is in this repo: the LaTeX source, the figure
builders, and the `results/*.json` artifacts the figures and numbers are read
from. No GPU, no model weights, no dataset download. A clean VM goes from
nothing to a compiled PDF in about five minutes, most of it spent letting
Tectonic fetch its TeX package bundle.

```bash
git clone git@github.com:AlexKay28/DLLM-as-Service.git
cd DLLM-as-Service/paper
make check      # confirm the toolchain resolved
make            # figures -> paper -> paper/v0/main.pdf
```

---

## 1. Toolchain

Three things: Python 3.10+ with matplotlib, Tectonic, and Poppler (only for
the page-count line in `make publish` and for eyeballing pages).

Tectonic is used rather than a full TeX Live because it is a single binary
that downloads exactly the packages this document asks for and caches them —
about 30 MB installed against 5 GB for MacTeX or `texlive-full`.

### Debian / Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip poppler-utils fonts-firasans curl
pip3 install --user "matplotlib>=3.5" numpy
curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
sudo mv tectonic /usr/local/bin/
```

The installer drops a `tectonic` binary into the current directory; moving it
onto `PATH` is the only manual step. `tectonic` is also packaged on Debian 13+
and Ubuntu 24.04+ (`apt-get install tectonic`) if you prefer the distro build.

### Fedora / RHEL

```bash
sudo dnf install -y python3 python3-matplotlib poppler-utils fira-sans-fonts
curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
sudo mv tectonic /usr/local/bin/
```

### macOS

```bash
brew install tectonic poppler
python3 -m pip install --user "matplotlib>=3.5" numpy
```

### With `uv` (the repo already carries a lockfile)

`matplotlib` is a top-level dependency in `pyproject.toml`, so the paper build
needs nothing beyond the base environment:

```bash
uv sync                       # from the repo root
cd paper && make PYTHON="uv run python"
```

---

## 2. Build

```bash
cd paper

make check      # versions + which font the figures resolved to
make figures    # rebuild all 9 figures from ../results/*.json
make paper      # compile main.tex -> main.pdf, report overfull boxes
make verify     # assert every cited number against its artifact
make publish    # snapshot to paper/v0/main.pdf
make            # all of the above except verify
make clean      # remove .aux/.log/.out (keeps main.pdf)
```

Expected output from a good build:

```
>> building figures
  wrote fig1_cost_shapes
  ... (9 total)
>> compiling main.tex
>> 0 overfull box(es), 0 undefined reference(s)
>> published paper/v0/main.pdf (21 pages)
```

`make verify` should end with `Errors: 0`. It also prints ~109 warnings; those
are pre-existing gaps in provenance blocks on older Campaign-A/D artifacts,
not build failures.

### Cutting a new version

Released snapshots live in `paper/<VERSION>/main.pdf`. Bump the version rather
than overwriting the last one:

```bash
make VERSION=v1 publish
```

Then update the version line in `main.tex` (`\papertitle`'s fifth argument) so
the cover page and the directory agree.

---

## 3. What the build actually reads

```
paper/
├── main.tex          entry point; \input's style, macros, and 10 sections
├── style.tex         the design system — fonts, colours, section heads, boxes
├── macros.tex        every number the paper cites, each traced to an artifact
├── sections/*.tex    prose; contains no numerals, only macros
├── references.bib
└── figures/
    ├── style.py      shared palette, type scale, and mark specs
    └── make_figures.py   reads ../../results/*.json, writes fig*.pdf + fig*.png
```

The rule that keeps text and figures consistent: **no numeral is typed into a
section file**. If it is a measurement it is a macro in `macros.tex`, and the
comment on its line names the artifact it was read from. `make_figures.py`
prints every value it plotted, so the two can be diffed after any change to
`results/`.

---

## 4. Gotchas we hit, so you don't have to

**Tectonic's first run needs network and takes a few minutes.** It fetches the
TeX bundle on demand and caches it under `~/.cache/Tectonic`. Later builds take
seconds. On an air-gapped VM, prime the cache on a connected machine and copy
that directory across.

**Figures fall back to DejaVu Sans on a bare VM.** `figures/style.py` prefers
Fira Sans (the paper's heading face), then Helvetica Neue, then Arial, then
matplotlib's bundled DejaVu Sans — so the build never fails on a missing font,
it just looks slightly different. Install `fonts-firasans` to match the text
typographically. `make check` reports which one resolved; the committed
figures were built with Helvetica Neue on macOS.

**`verify_numbers.py` needs Python 3.10+.** It imports
`models/baselines/provenance.py`, which uses PEP 604 unions (`X | dict`), and
those are a syntax error on 3.9. macOS system Python is 3.9, so pass an
explicit interpreter: `make PYTHON=python3.12 verify`.

**Do not add `fontawesome5`.** It is not in the Tectonic bundle and the
compile fails hard. If you want icons, use a package that ships with TeX Live.

**Package order in `style.tex` matters in two places.** `tcolorbox` loads
`xcolor` itself, so loading `xcolor` with options after it raises an option
clash; and `newtxmath` already defines `\Bbbk`, so `amssymb` must be preceded
by `\let\Bbbk\relax`. Both are already handled — just do not reorder those
lines.

**Figure PDFs carry a creation timestamp.** Re-running `make figures` without
changing anything still produces byte-different files, so `git status` will
show all nine as modified. Check the rendering actually changed before
committing that churn; `git checkout -- figures/` discards it.

---

## 5. Reproducing the figures against different results

The figure builders take no arguments and read `results/` directly, so
regenerating after a new experiment run is just `make figures`. Two guards
exist to catch drift:

- `fig9_deployment` re-reads the AR-1.5B forced-length p50s and asserts them
  against the values it plots, so the build fails loudly if that artifact
  moves.
- The fitted constants live in one `LAWS` dict at the top of
  `make_figures.py` and must match `macros.tex`. If you refit a law, change
  both.
