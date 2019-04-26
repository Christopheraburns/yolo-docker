FROM ubuntu:latest
MAINTAINER burnsca@amazon.com


# Update base
RUN apt-get update

# Install lots-o-dependencies
RUN apt-get install -y python3 python3-pip libxdamage1 libxext6 libpixman-1-0 libfreetype6 
RUN apt-get install -y libzvbi-common libzvbi0 libxvidcore4 libxcb-shm0 libxrandr2 libxkbcommon0 libxinerama1
RUN apt-get install -y libxcb-render0 libxi6 libxfixes3 libxcursor1 libxcomposite1 libatk1.0-0 libatk-bridge2.0.0 
RUN apt-get install -y libdrm2 libatspi2.0-0 libavcodec57 libgssapi-krb5-2 libavformat57 libavutil55 libbluray2
RUN apt-get install -y libcuda1-384


# move contents to app dir 
COPY . /app
WORKDIR /app

# Install python package dependencies
RUN pip3 install -r requirements.txt 

# Set the LIB path
ENV LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Expose port 5000 for Flask
EXPOSE 5000

# Start the app
CMD python3 ./darknet_app.py

