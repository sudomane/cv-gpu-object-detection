import cv2
import json

import argparse

def getArgs():
    parser = argparse.ArgumentParser(description='Create a video from the JSON file outputted by the object detection C++ pipeline.')

    parser.add_argument('-o', '--output_filename', type=str, default='output_video.avi')
    parser.add_argument('-j', '--json_path', type=str, default='build/bbox.json')
    parser.add_argument('-f', '--fps', type=int, default=10)

    args = parser.parse_args()

    return args

def loadJSON(json_path : str):
    with open(json_path, "r") as fp:
        data = json.load(fp)

    return data

def toVideo(frames: list, output_path: str, fps: int):
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()

def main(args):

    bbox_data = loadJSON(args.json_path)

    frames = []

    for img_path, rect_shape in bbox_data.items():
        img   = cv2.imread(img_path)
        shape = (*rect_shape[0], *rect_shape[1])

        cv2.rectangle(img, (shape[2], shape[3]), (shape[1], shape[0]), (0, 0, 255), 3)

        frames.append(img)

    toVideo(frames, args.output_filename, args.fps)

if __name__ == "__main__":
    args = getArgs()
    main(args)