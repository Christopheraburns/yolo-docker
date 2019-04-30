FROM nvidia/cuda:9.2-base-ubuntu18.04
MAINTAINER burnsca@amazon.com


# Update base
RUN apt-get update

# Set Environment variables here to avoid a dialog
ENV TZ=America/Chicago

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y python3-opencv
RUN apt-get install -y nginx ca-certificates wget unzip libopenblas-base


# move contents to app dir
COPY yolo /opt/program
WORKDIR /opt/program

# Download weigths file
RUN wget https://s3.amazonaws.com/cardbot-data/aces_4000.weights

# Download, move and unzip CudaLibs
RUN wget https://s3.amazonaws.com/cardbot-data/cuda_lib64.zip && mv cuda_lib64.zip /usr/local/cuda/lib64/cuda_lib64.zip &&
unzip /usr/local/cuda/lib64/cuda_lib64.zip

#  move and unzip x86_64-linux-gnu.zip and
RUN mv x86_64-linux-gnu.zip /usr/lib/x86_64-linux-gnu/x86_64-linux-gnu.zip &&
unzip /usr/lib/x86_64-linux-gnu/x86_64-linux-gnu.zip


ENV PATH="/opt/program:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Install python package dependencies
RUN pip3 install flask boto3 gputil gevent gunicorn







