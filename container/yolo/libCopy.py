import os
from shutil import copyfile


# This code assumes you have downloaded req_libs.zip from https://s3.amazonaws.com/cardbot-data/libs.zip
# and unzipped to a directory within this app dir called "deleteMe"

lib_names = []
lib_path = []


with open("depend.csv", "r") as f:
    for line in f:
        split = line.split(",")
        lib_names.append(str(split[0]))
        lib_path.append(str(split[1].strip()))

# create directories if necessary and copy files (if they don't already exist)

libraries = zip(lib_path, lib_names)
for body, head in libraries:
    # get just the directory of the file
    path = body[: body.index("/", 10)]
    if not os.path.isdir(path):
        # create the dir
        os.mkdir(path)

    # Does the file already exist?
    if not os.path.isfile(body):
        # copy the file
        copyfile("deleteMe/" + head, body)



