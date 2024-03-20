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

    for img_path, dim in bbox_data.items():
        img   = cv2.imread(img_path)

        if (dim != []):
            top_left  = (dim[0][0], dim[1][1])
            bot_right = (dim[0][1], dim[1][0])
            print(f"Top left: {top_left} | Bot: {bot_right}")
            cv2.rectangle(img, top_left, bot_right, (0, 0, 255), 3)

        frames.append(img)

    toVideo(frames, args.output_filename, args.fps)

if __name__ == "__main__":
    args = getArgs()
    main(args)