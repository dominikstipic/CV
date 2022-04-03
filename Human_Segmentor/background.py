import cv2
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.piramid_swiftnet.model import PiramidSwiftnet

def parse_args():
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("-weights", default="weights/1.pth", help="Weights")
    parser.add_argument("-human", default="figs/human.jpeg", help="Human")
    parser.add_argument("-background", default="figs/1.jpeg", help="Photo")
    args = parser.parse_args()

    weights_path = args.weights
    weights = torch.load(weights_path, map_location="cpu")

    model = PiramidSwiftnet(2)
    model.load_state_dict(weights)
    model.eval_state()

    human_photo = cv2.imread(args.human)
    background_photo = cv2.imread(args.background)
    return model, human_photo, background_photo

def resize(img: np.array, size: int):
    return cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)

def unnormalize(img, mean, std):
    for i in range(3):
        img[:,:,i] = img[:,:,i]*std[i] + mean[i]
    return img 

def normalize(img, mean, std):
    img = img.astype(np.float32)
    for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean[i])/std[i]
    return img

def prediction(img, model):
    pred = model(img.unsqueeze(0)).squeeze(0).argmax(0).numpy()
    pred = pred.astype(np.uint8)
    return pred

def channel_last(img):
    img = np.transpose(img, (1,2,0))
    img = np.ascontiguousarray(img)
    return img

def channel_first(img):
    img = np.transpose(img, (2,0,1))
    img = np.ascontiguousarray(img)
    return img 

def cut_human(human_photo: np.array, model: PiramidSwiftnet):
    norm_human = normalize(human_photo, tf_mean, tf_std)
    norm_human = torch.from_numpy(channel_first(norm_human))
    segmented = prediction(norm_human, model)

    human_photo  = channel_first(human_photo)
    cutted_human = segmented * human_photo
    cutted_human = unnormalize(cutted_human, tf_mean, tf_std)
    cutted_human = channel_last(cutted_human)
    cutted_human = cutted_human.astype(np.uint8)
    return cutted_human, segmented

def stick_images(background_photo: np.array, cutted_human_photo: np.array, segmented: np.array):
    segmented = segmented > 0
    for i in range(3):
        cutted_human_photo_channel = cutted_human_photo[:,:,i]
        background_photo_channel   = background_photo[:,:,i]
        human_pixels = cutted_human_photo_channel[segmented]
        background_photo_channel[segmented] = human_pixels
    return background_photo

#######################

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

SIZE = 400

def process(model, human_photo, background_photo):
    cutted_human_photo, segmentation_map = cut_human(human_photo, model)
    result = stick_images(background_photo.copy(), cutted_human_photo, segmentation_map)
    return result

if __name__ == "__main__":
    model, human_photo, background_photo = parse_args()
    human_photo, background_photo = resize(human_photo, SIZE), resize(background_photo, SIZE)
    result = process(model, human_photo, background_photo)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()




