FROM nvidia/cuda
MAINTAINER burnsca@amazon.com


# Update base
RUN apt-get update

# Set Environment variables here to avoid a dialog
ENV TZ=America/Chicago

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y python3-opencv
RUN apt-get install -y nginx ca-certificates wget


# move contents to app dir
COPY yolo /opt/program
WORKDIR /opt/program

RUN wget https://s3.amazonaws.com/cardbot-data/aces_4000.weights

# Copy all the libraries from /usr/local/lib to the container
WORKDIR /usr/local/lib

COPY lib* /usr/local/lib

WORKDIR /opt/program


ENV PATH="/opt/program:${PATH}"
ENV LD_LIBRARY_PATH'/usr/local/lib:${LD_LIBRARY_PATH}

# Install python package dependencies
RUN pip3 install flask boto3 gputil gevent gunicorn







