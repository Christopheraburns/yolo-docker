FROM nvidia/cuda:9.2-base-ubuntu16.04
MAINTAINER burnsca@amazon.com


# Update base
RUN apt-get update

# Set Environment variables here to avoid a dialog
ENV TZ=America/Chicago

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get install -y python3 python3-pip
#RUN apt-get install -y python3-opencv
RUN apt-get install -y nginx ca-certificates wget unzip libopenblas-base
RUN apt-get install -y libpangox-1.0-0 libv4l2rds0 libswresample-ffmpeg1 libva1 libx265-79
RUN apt-get install -y libx264-148 libwebp5 libvpx3 libschroedinger-1.0-0 libopenjpeg5 libmodplug1
RUN apt-get install -y libbluray1 libglu1-mesa libxmu6

# move contents to app dir
COPY yolo /opt/program
WORKDIR /opt/program


# Download weigths file
RUN wget https://s3.amazonaws.com/cardbot-data/aces_4000.weights

# Download, and unzip CudaLibs
RUN wget https://s3.amazonaws.com/cardbot-data/cuda_lib64.zip \
    && unzip cuda_lib64.zip -d /usr/local/cuda/lib64/


#  unzip x86_64-linux-gnu.zip
RUN unzip x86_64-linux-gnu.zip -aoa -d /usr/lib/x86_64-linux-gnu/


ENV PATH="/opt/program:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Install python package dependencies
RUN pip3 install flask boto3 gputil gevent gunicorn







