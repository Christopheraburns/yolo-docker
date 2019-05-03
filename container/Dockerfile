FROM nvidia/cuda:10.1-runtime-ubuntu18.04


# Install utils
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         python3-pip \
         git \
         nginx \
         ca-certificates \
         unzip \
    && rm -rf /var/lib/apt/lists/*


# pip Install dependencies
RUN pip3 install flask boto3 gevent gunicorn && \
        rm -rf /root/.cache

ENV PATH="/opt/program:$PATH


# Copy Source code and set WORKDIR
COPY yolo /opt/program
WORKDIR /opt/program


# wget model weights
RUN wget https://s3.amazonaws.com/cardbot-data/aces_4000.weights && \
    wget https://s3.amazonaws.com/cardbot-data/bodyparts.zip


# unzip bodyparts
RUN mkdir deleteMe && \
    unzip bodyparts.zip -d deleteMe

# copy files and delete old
RUN python3 libCopy.py && \
    rm -rf bodyparts.zip && \
    rm -rf deleteMe
