import time
import argparse
import sys

import cv2
import torch
import numpy as np
from importlib import import_module

import background as bckg 

def get_object_from_standard_name(std_name: str):
    pckg_parts = std_name.split(".")
    pckg_name, cls_name = ".".join(pckg_parts[:-1]), pckg_parts[-1]
    try:
        pckg = import_module(pckg_name)
        obj = getattr(pckg, cls_name)
    except ModuleNotFoundError:
        print(f"Could not find specified object in python search path: {std_name}")
        sys.exit(1)
    return obj

def parse_args():
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("-weights", default="1.pth", help="Weights")
    parser.add_argument("-model", default="piramid_swiftnet.model.PiramidSwiftnet", help="Model")
    parser.add_argument("-background", default="figs/1.jpeg", help="Photo")
    args = parser.parse_args()

    weights_path = f"weights/{args.weights}"
    weights = torch.load(weights_path, map_location="cpu")

    MODELS_PATH = "src.models"
    std_name = f"{MODELS_PATH}.{args.model}"  
    model_obj = get_object_from_standard_name(std_name)
    model = model_obj(2)
    
    model.load_state_dict(weights)
    model.eval_state()

    background_photo = cv2.imread(args.background)
    return model, background_photo

tf_mean = np.array([
            119.12654113769531,
            112.16575622558594,
            108.71101379394531
            ])
tf_std = np.array([
            67.10334777832031,
            66.58047485351562,
            69.03285217285156
            ])

SIZE = 200

######################

def delist(xs):
    ys = torch.tensor([])
    for x in xs:
        ys = torch.hstack([ys, x.flatten()])
    return ys

import matplotlib.pyplot as plt
weights_path = f"weights/1.pth"
w1 = torch.load(weights_path, map_location="cpu")
w2 = torch.load("weights/2.pth", map_location="cpu")

w1 = delist(list(w1.values()))
w2 = delist(list(w2.values()))

plt.subplot(1,2,1)
plt.hist(w1)
plt.subplot(1,2,2)
plt.hist(w2)
plt.show()
quit()

model, background_photo = parse_args()
background_photo = bckg.resize(background_photo, SIZE)
vid = cv2.VideoCapture(0)
times = np.array([])
while(True):
    _,frame = vid.read()
    frame = bckg.resize(frame, SIZE)
    start = time.perf_counter()
    frame = bckg.process(model, frame, background_photo)
    end = time.perf_counter()
    frame = bckg.resize(frame, 2*SIZE)
    cv2.imshow('frame', frame)

    t_sec = end - start
    print(f"time={t_sec} sec")
    times = np.append(times, t_sec)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

mean_time = times.mean()
fps = 1/mean_time
print(f"FPS={fps}")

vid.release()
cv2.destroyAllWindows()
