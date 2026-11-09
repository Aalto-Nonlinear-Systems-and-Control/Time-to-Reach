# check=skip=FromPlatformFlagConstDisallowed
# (the platform is fixed on purpose: the base images and dReal3's dependencies are x86-64 only)

# dReach is part of dReal3 (github.com/dreal/dreal3), whose last commit is 7f31797b (2021-07-08).
#
# Build:   docker build --platform linux/amd64 -t dreach:dreal3-7f31797b tools/dreach
#          (add --build-arg JOBS=8 to compile with more parallel jobs; the default 4 suits 8 GB of RAM)
#
# Reproducible inputs: both base images are pinned by digest; dReal3 is pinned by commit and its
# CMake pins every third-party dependency (IBEX, CAPD, filib, ...) by commit; the OCaml packages
# come from the opam-repository snapshot inside the pinned build image (2023-06-22), at fixed
# versions; Ubuntu 18.04 is end-of-life, so its package archive no longer changes.
#
# Two stages: the build stage holds compilers and sources (several GB); the runtime image holds
# only dReach, dReal and bmc, the shared libraries they link to, cpp and time.
# The images have no arm64 variant, so on Apple Silicon dReach runs under emulation.

ARG DREAL3_COMMIT=7f31797b728ba54b0cb0000dfae537006da8d191

FROM --platform=linux/amd64 docker.io/ocaml/opam:ubuntu-18.04-opam@sha256:055d7f71ecb331811742d31993edc71e9f561fe59c0ab623265f9ddf593a91e0 AS build

USER root
RUN apt-get -y update && \
    apt-get -y install -qq autoconf automake bison flex git libtool make pkg-config texinfo \
    python3-dev gcc g++ cmake build-essential \
    libgmp-dev libgmpxx4ldbl libmpfr-dev libboost-all-dev libeigen3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    (command -v python || ln -s /usr/bin/python3 /usr/bin/python)

# dReal (C++); its CMake clones the pinned third-party dependencies
ARG DREAL3_COMMIT
RUN git clone https://github.com/dreal/dreal3.git /usr/app/dreal && git -C /usr/app/dreal checkout ${DREAL3_COMMIT}
WORKDIR /usr/app/dreal/build/release
ARG JOBS=4
RUN cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ../../src && \
    make -j"${JOBS}"

# OCaml packages for the dReach front end (bmc), from the image's own opam-repository snapshot
USER opam
RUN opam switch create default ocaml-base-compiler.4.14.0 && \
    opam install -y ocamlfind.1.9.6 batteries.3.6.0 oasis.0.4.11 && \
    opam list > /home/opam/opam-packages.txt

USER root
WORKDIR /usr/app/dreal/tools
RUN eval $(opam env --root=/home/opam/.opam --switch=default --set-switch) && make

# Collect what the runtime needs. bin/dReach finds its own path with `python -c ... realpath`;
# readlink -f does the same, so the runtime image needs no Python. Libraries that ubuntu:18.04
# already ships (libc, libm, libstdc++, ...) are left out.
RUN set -eux; \
    mkdir -p /opt/dreach/bin /opt/dreach/lib; \
    cp -L /usr/app/dreal/bin/dReach /usr/app/dreal/bin/dReal /usr/app/dreal/bin/bmc /opt/dreach/bin/; \
    sed -i 's|^SCRIPT_PATHNAME=.*|SCRIPT_PATHNAME=$(readlink -f "$0")|' /opt/dreach/bin/dReach; \
    grep -qF 'SCRIPT_PATHNAME=$(readlink -f "$0")' /opt/dreach/bin/dReach; \
    strip /opt/dreach/bin/dReal /opt/dreach/bin/bmc; \
    ldd /opt/dreach/bin/dReal /opt/dreach/bin/bmc \
        | awk '$2 == "=>" && $3 ~ /^\// {print $3}' | sort -u \
        | grep -vE '/(libc|libm|libdl|libpthread|librt|libgcc_s|libstdc\+\+)\.so' \
        | xargs -r cp -L -t /opt/dreach/lib/; \
    git -C /usr/app/dreal rev-parse HEAD > /opt/dreach/DREAL3_COMMIT; \
    cp /home/opam/opam-packages.txt /opt/dreach/

FROM --platform=linux/amd64 docker.io/library/ubuntu:18.04@sha256:152dc042452c496007f07ca9127571cb9c29697f42acbfad72324b2bb2e43c98

ARG DREAL3_COMMIT
LABEL org.opencontainers.image.title="dReach (dReal3 ${DREAL3_COMMIT})" \
      org.opencontainers.image.source="https://github.com/dreal/dreal3"

# bin/dReach preprocesses models with cpp; the benchmark runner times dReach with /usr/bin/time
RUN apt-get -y update && \
    apt-get -y install -qq --no-install-recommends cpp time && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/dreach /opt/dreach
ENV PATH="/opt/dreach/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/dreach/lib"

COPY selftest /opt/dreach/selftest
RUN sh /opt/dreach/selftest/selftest.sh

WORKDIR /work
