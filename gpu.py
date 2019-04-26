import GPUtil


gpu = GPUtil.getGPUs()

if gpu:
    print(gpu[0].name)

print("stophere")