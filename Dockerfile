FROM ubuntu:latest
MAINTAINER burnsca@amazon.com


# Update base
RUN apt-get update

# Set Environment variables here to avoid a dialog
ENV TX=America/Chicago

RUN lb -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y python3-opencv


# move contents to app dir 
COPY . /app
WORKDIR /app

# Install python package dependencies
RUN pip3 install -r requirements.txt 


# Expose port 5000 for Flask
EXPOSE 5000

# Start the app
CMD python3 ./yolov3_app.py



