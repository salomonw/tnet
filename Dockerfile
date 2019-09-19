# Dockerfile
#
# Parent iamge
FROM ubuntu:16.04

# Required installations
RUN apt-get -y update && apt-get install -y \
  python3 \
  python3-pip \
  make \
  unzip \
  mercurial \
  wget \
  libgmp3-dev \
  vim


# Python libraries
RUN apt-get -y update
RUN apt-get -y upgrade
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install julia

RUN pip3 install networkx


# Julia dependencies
# Install Julia packages in /opt/julia instead of $HOME
ENV JULIA_DEPOT_PATH=/opt/julia
ENV JULIA_PKGDIR=/opt/julia
ENV JULIA_VERSION=1.0.0

RUN mkdir /opt/julia-${JULIA_VERSION} && \
    cd /tmp && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    echo "e0e93949753cc4ac46d5f27d7ae213488b3fef5f8e766794df0058e1b3d2f142 *julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | sha256sum -c - && \
    tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
    rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz
RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

# Install PyJulia requirement PyCall
RUN julia -e 'import Pkg; Pkg.update()' && \
	julia -e 'import Pkg; Pkg.add("LightGraphs")' && \
	julia -e 'import Pkg; Pkg.add("Optim")' && \
	julia -e 'import Pkg; Pkg.add("BinDeps")' && \
	julia -e 'import Pkg; Pkg.add("NPZ")' $$ \
	julia -e 'import Pkg; Pkg.add("JuMP")' $$ \	
	julia -e 'import Pkg; Pkg.add("Ipopt")' $$ \	
    julia -e 'import Pkg; Pkg.add("PyCall")'

# Install PyJulia
RUN python3 -m pip install julia && \
    python-jl --version

# TODO: install gurobi 

# Add libraries
COPY . /tnet
WORKDIR /tnet



