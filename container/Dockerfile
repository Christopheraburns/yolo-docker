FROM nvidia/cuda:10.1-runtime-ubuntu18.04


# Install utils
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         python3-pip \
         nginx \
         ca-certificates \
         unzip \
    && rm -rf /var/lib/apt/lists/*


# pip Install dependencies
RUN pip3 install \
        setuptools \
        flask \
        gevent \
        gunicorn \
        requests \
        pillow && \
   rm -rf /root/.cache

ENV PATH="/opt/program:${PATH}"


# Copy Source code and set WORKDIR
COPY yolo /opt/program
WORKDIR /opt/program


# wget model weights
RUN wget https://s3.amazonaws.com/cardbot-data/aces_4000.weights && \
    wget https://s3.amazonaws.com/cardbot-data/req_libs.zip


# unzip bodyparts
RUN mkdir deleteMe && \
    unzip req_libs.zip -d deleteMe && \
    python3 libCopy.py && \
    rm -rf req_libs.zip && \
    rm -rf delteMe

