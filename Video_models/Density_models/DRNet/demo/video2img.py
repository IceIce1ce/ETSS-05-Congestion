import cv2
import os
import argparse

def getFrame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        flag, frame = cap.retrieve()
        if not flag:
            continue
        else:
            numFrame += 1
            print('Frame:', numFrame)
            newPath = os.path.join(svPath, str(numFrame) + ".jpg")
            cv2.imencode('.jpg', frame)[1].tofile(newPath)
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='demo.mp4')
    parser.add_argument('--output_dir', type=str, default='data/demo/img1')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    getFrame(args.input_dir, args.output_dir)