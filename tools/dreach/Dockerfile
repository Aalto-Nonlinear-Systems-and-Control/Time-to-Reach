
FROM --platform=linux/amd64 docker.io/ocaml/opam:ubuntu-18.04-opam

USER root

# Install required packages
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install -qq autoconf automake bison flex git libtool make pkg-config texinfo \
    python3-dev python3-pip wget curl gcc g++ cmake build-essential \
    libgmp-dev libgmpxx4ldbl libmpfr-dev libboost-all-dev libeigen3-dev \
    bubblewrap && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dReal/dReach 3
RUN mkdir -p /usr/app
WORKDIR /usr/app
RUN git clone https://github.com/dreal/dreal3.git dreal
WORKDIR /usr/app/dreal/build/release
RUN cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ../../src && \
    make

# Add dReal binaries to PATH
ENV PATH="/usr/app/dreal/build/release/bin:${PATH}"

# Switch to opam user for OPAM operations
USER opam
WORKDIR /home/opam

# Initialize OPAM and install OCaml packages
RUN opam init --disable-sandboxing -y && \
    opam switch create default ocaml-base-compiler.4.14.0 && \
    eval $(opam env) && \
    opam update && \
    opam install -y ocamlfind batteries oasis

# Switch back to root and compile dReach tools
USER root
WORKDIR /usr/app/dreal/tools

# Set up environment and compile tools
RUN eval $(opam env --root=/home/opam/.opam) && make

# Add dReach tools to PATH
ENV PATH="/usr/app/dreal/tools:${PATH}"

# Set working directory back to app
WORKDIR /usr/app
