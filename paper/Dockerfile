# Reproducible paper build. Nothing here needs a GPU, model weights, or the
# raw corpora — the figures and every cited number are read from results/.
#
#   docker build -f paper/Dockerfile -t dllm-paper .
#   docker run --rm -v "$PWD/out:/out" dllm-paper
#
# The compiled PDF lands in ./out/main.pdf. To iterate on the source without
# rebuilding the image, mount the repo over it:
#
#   docker run --rm -v "$PWD:/repo" -w /repo/paper dllm-paper make
#
# Build context is the repo root, not paper/, because the figure builders read
# ../results.

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/root

# fonts-firasans matches the paper's heading face, so figures render with the
# same sans as the text instead of falling back to DejaVu.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-matplotlib python3-numpy \
        poppler-utils fonts-firasans \
        make curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Tectonic: one binary, fetches the TeX packages this document actually asks
# for rather than installing all of TeX Live.
RUN curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh \
    && mv tectonic /usr/local/bin/tectonic \
    && tectonic --version

WORKDIR /repo
COPY . /repo

# Warm the TeX bundle cache at build time so `docker run` works offline and
# takes seconds rather than minutes.
RUN cd paper && tectonic -X compile main.tex --keep-logs && make clean

WORKDIR /repo/paper

# Default: rebuild everything from the artifacts and drop the PDF in /out if
# that path is mounted.
CMD make figures paper \
    && if [ -d /out ]; then cp main.pdf /out/ && echo ">> wrote /out/main.pdf"; fi \
    && make clean
