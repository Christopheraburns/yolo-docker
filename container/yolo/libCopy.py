import os
from shutil import copyfile


pile_of_heads = []
pile_of_bodies = []


INSTALL = True


with open("depend.csv", "r") as f:
    for line in f:
        split = line.split(",")
        pile_of_heads.append(str(split[0]))
        pile_of_bodies.append(str(split[1].strip()))



if INSTALL:
    # create directories if necessary and copy files (if they don't already exist)
    stop=""
else:
    # Zip existing file
    if not os.path.isdir("bodyparts"):
        os.mkdir("bodyparts")

    bodyparts = zip(pile_of_heads, pile_of_bodies)
    for head, body in bodyparts:
        # Find the file and copy to a local directory
        copyfile(body, 'bodyparts/' + head)






