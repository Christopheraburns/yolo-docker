Ongoing work to fit C++ yolo model into Sagemaker container for hosting


download aces_4000.weights from s3://cardbot-data/

for local development use libyolo_dummy.so - this was compiled on Ubuntu 18 - NO GPU and should work on almost any Ubuntu machine

Switch to libyolo_volta on P3 (make sure it is a V100 GPU and not the P100)

