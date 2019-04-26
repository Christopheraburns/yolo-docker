#!python3

# pylint: disable=R, W0401, W0614, W0703
import timeit
from ctypes import *
import random
from flask import Flask
import boto3
import GPUtil


app = Flask(__name__)

start = timeit.default_timer()

try:
    gpus = GPUtil.getGPUs()
    if gpus:
        if 'V100' in str(gpus[0].name):
            lib = CDLL("./libyolo_volta.so", RTLD_GLOBAL)
        else: # Not a volta core - use the NOGPU option
            lib = CDLL("./libyolo_dummy.so", RTLD_GLOBAL)
    else:
        lib = CDLL("./libyolo_dummy.so", RTLD_GLOBAL)
except:
    lib = CDLL("./libyolo_dummy.so", RTLD_GLOBAL)


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


detect_time = None
stop = timeit.default_timer()
pre_load_time = stop-start

start = timeit.default_timer()
thresh = 0.25

# HardCode these variables for now
config_path = "aces.cfg"
weight_path = "aces_4000.weights"
meta_path = "aces.data"
net_main = load_net_custom(config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
meta_main = load_meta(meta_path.encode("ascii"))
image_path = "test.jpg"

# Load the class names
with open("aces.names") as namesFH:
    names_list = namesFH.read().strip().split("\n")


stop = timeit.default_timer()
param_load_time = stop-start



def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        name_tag = meta.names[i]
        res.append((name_tag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    """
    Performs the meat of the detection
    """
    global detect_time
    dstart = timeit.default_timer()
    # pylint: disable= C0321
    im = load_image(image, 0, 0)

    ret = detect_image(net, meta, im, thresh, hier_thresh, nms)
    free_image(im)
    
    dstop = timeit.default_timer()
    detect_time = dstop - dstart
    return ret


def detect_image(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):

    # import cv2
    # custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    # custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    # custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    # import scipy.misc
    # custom_image = scipy.misc.imread(image)
    # im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)
    num = c_int(0)

    pnum = pointer(num)

    predict_image(net, im)

    # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, 0) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)

    num = pnum[0]

    if nms:
        do_nms_sort(dets, num, meta.classes, nms)

    res = []

    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                name_tag = meta.names[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
                #res.append(name_tag)

    inferences = "{ 'inferences':"
    for i in res:
        inferences = inferences + "{"
        index = 0
        SR = None
        Conf = None
        Coords = None
        for section in i:  # should be three - SUIT + RANK, CONFIDENCE, Coords
            index += 1
            if index is 1:  # SUIT+RANK
                SR = section
                inferences = inferences + "'SuiteRank':'" + SR.decode("utf-8") + "',"
            elif index is 2:
                Conf = section
                inferences = inferences + "'Confidence':'" + str(Conf) + "',"
            else:
                Coords = section
                inferences = inferences + "'Coords':'" + str(Coords) + "'}"
                index = 0
    inferences = inferences + "}"

    return inferences



# Verification function
@app.route('/test')
def test():
    # Return the detection results from the test image to verify functionality
    return (detect(net_main, meta_main, image_path.encode("ascii"), thresh))


# Accept an S3 path to the image to
@app.route('/<s3Path>', methods=['PUT'])
def index(s3Path):

    s3 = boto3.client('s3')


    # Return the detection results
    return(detect(net_main, meta_main, image_path.encode("ascii"), thresh))




if __name__ == "__main__":
    app.run(host= '0.0.0.0')
