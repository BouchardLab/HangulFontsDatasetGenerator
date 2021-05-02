FROM debian:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /home/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y build-essential gfortran wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

SHELL [ "/bin/bash", "--login", "-c" ]

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /home/conda && \
    rm ~/miniconda.sh && \
    /home/conda/bin/conda clean -tipsy && \
    ln -s /home/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /home/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Initializes conda
ENV PATH=/home/conda/bin:$PATH
# Install
RUN conda update -n base -c defaults conda
RUN conda install python=3.7
RUN conda install numpy scipy pandas scikit-image h5py

ADD HangulFontsDatasetGenerator /home/HanGulFontsDatasetGenerator
RUN pip install -e /home/HanGulFontsDatasetGenerator

# Setup entrypoint
ADD entrypoint.sh /home/entrypoint.sh
RUN chmod 777 /home/entrypoint.sh
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT [ "/usr/bin/tini", "--", "/home/entrypoint.sh" ]
CMD [ "/bin/bash" ]
