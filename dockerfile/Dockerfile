FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]

# Set up a non-root user, `user`, with a group, `group`
ENV HOME=/home
RUN mkdir -p $HOME

# Install pip
RUN apt-get update && apt-get dist-upgrade -y
RUN apt-get update && apt-get install -y less wget rsync git

# Install conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh --directory-prefix=$HOME
RUN /bin/bash $HOME/Miniconda3-4.7.12-Linux-x86_64.sh -bp /miniconda3
RUN rm $HOME/Miniconda3-4.7.12-Linux-x86_64.sh
ENV PATH /miniconda3/bin:$PATH

RUN conda config --prepend channels conda-forge
RUN conda config --append channels matsci
RUN conda install \
    python=3.6 \
    numpy=1.19.2 \
    ase=3.21.1 \
    cffi==1.14.4 \
    dask==2.30.0
RUN conda clean -ity

RUN pip install torch==1.6.0
RUN pip install git+https://github.com/yilinyang1/nn_optimizer.git
