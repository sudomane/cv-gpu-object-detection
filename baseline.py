#!/bin/usr/ python3

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, gaussian_filter
from skimage.morphology import closing, opening

import argparse

def get_img(path : str) -> np.ndarray:
    if (not(os.path.isfile(path))):
        raise RuntimeError(f"File {path} does not exist.")
    
    image = cv2.imread(path)
    return image

def get_largestCC(image : np.ndarray) -> np.ndarray:
    mask, n_labels = label(image)

    if (n_labels == 0):
        raise RuntimeError("Found no CC in image.")
    
    max_CC  = np.argmax(np.bincount(mask.flatten())) + 1
    mask_CC = np.where(mask == max_CC, 1, 0)

    return mask_CC

def to_gray(image : np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def blur_image(image : np.ndarray, sigma : int = 5) -> np.ndarray:
    blurred_image = gaussian_filter(image, sigma=sigma)
    return blurred_image

def get_bbox(mask : np.ndarray) -> np.ndarray:
    indices = np.where(mask)
    
    # No object detected
    if (indices[0].size == 0):
        return None
    
    min_x, min_y = np.min(indices[1]), np.min(indices[0])
    max_x, max_y = np.max(indices[1]), np.max(indices[0])
    
    return ((min_x, min_y), (max_x, max_y))
  
def binarize(image : np.ndarray, threshold : int = 0.5):
    if (np.max(image) != 1):
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image = np.where(image > threshold, 1, 0)

    return image

def detection_pipeline(ref_img : np.array, img : np.array, sigma : int = 5, bin_threshold : int = 0.5) -> np.ndarray:
    diff    = img - ref_img
    gray    = to_gray(diff)
    blur    = blur_image(gray, sigma=sigma)
    morph   = opening(closing(blur))
    bin_img = binarize(morph, threshold=bin_threshold)
    mask    = get_largestCC(bin_img)
    bbox    = get_bbox(mask)

    return bbox

# -------------

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--img",     help="New image with object on frame.")
    parser.add_argument("--width",   help="Image width.", default=1280, type=int)
    parser.add_argument("--height",  help="Image height.", default=720, type=int)
    parser.add_argument("--sigma",   help="Gaussian sigma size.", default=5, type=int)
    parser.add_argument("--ref_img", help="Reference image to compare subsequent frames to.")
    parser.add_argument("--output",  help="File to write detection to.", default="output.png")
    parser.add_argument("--live",    help="Get images from camera.", default=True, action="store_true")

    args = parser.parse_args()

    return args

def camera_main(args):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
 
    if not(cap.isOpened()):
        print("Could not open camera.")
        exit()
    if (args.ref_img is None):
        _, ref_img = cap.read()
    else:
        ref_img = get_img(args.ref_img)

    while True:
        _, frame = cap.read()

        t1 = time.time()
        bbox = detection_pipeline(ref_img, frame, sigma=args.sigma, bin_threshold=0.75)
        t2 = time.time()

        delta = t2 - t1
        fps   = 1 / delta

        cv2.putText(frame, f"FPS: {fps:.3f}", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        
        if (not(bbox is None)):
            min_x, min_y = bbox[0][0], bbox[0][1]
            max_x, max_y = bbox[1][0], bbox[1][1]
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 5)
        
        cv2.imshow('Camera Input', frame)
        
        if (cv2.waitKey(1) & 0xFF == ord('s')):
            print("[INFO] Setting new reference image...")
            _, ref_img = cap.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(args):
    assert args.img is None or args.ref_img is None, "No images provided for object detection."

    img     = get_img(args.img)
    ref_img = get_img(args.ref_img)

    t1 = time.time()
    bbox = detection_pipeline(ref_img, img)
    t2 = time.time()

    output_file = args.output

    delta = t2 - t1

    print(f"Object detected in {delta:3f}s!")
    print(f"Saving file to {output_file}.png!")

    min_x, min_y = bbox[0][0], bbox[0][1]
    max_x, max_y = bbox[1][0], bbox[1][1]
    
    plt.imshow(img)
    plt.plot([min_x, max_x, max_x, min_x, min_x],
             [min_y, min_y, max_y, max_y, min_y],
             color='red', label='Bounding Box')
    
    plt.savefig(output_file)

if __name__ == "__main__":
    args = get_args()
    if (args.live):
        camera_main(args)
    else:
        main(args)