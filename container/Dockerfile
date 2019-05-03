FROM ubunbtu:18.04


# Install utils
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         python3-pip \
         git \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# pip Install dependencies
RUN pip3 install flask boto3 gevent gunicorn && \
        rm -rf /root/.cache

# Copy Source code and set WORKDIR
COPY yolo /opt/program
WORKDIR /opt/program


# wget model weights
RUN wget https://s3.amazonaws.com/cardbot-data/aces_4000.weights

# wget model dependencies



ENV PATH="/opt/program:${PATH}"

