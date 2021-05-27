import picture
import cv2 as cv
import argparse
import numpy as np


class Video:
    def __init__(self, args):
        self.args = args
        self.img = picture.Tester(self.args)

    def run(self, vi_path):
        cap = cv.VideoCapture(vi_path)
        if not cap.isOpened():
            print("open failed")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv.imshow('lr', frame)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb = self.img.run(rgb, False)
            frame = cv.cvtColor(np.array(rgb), cv.COLOR_RGB2BGR)
            cv.imshow('hr', frame)
            # if cv.waitKey(10) & 0xFF == ord('q'):
            #     break
        cap.release()
        cv.destroyAllWindows()


def main(args):
    v = Video(args)
    v.run(args.video_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Input a lr_video to get a hr_video")
    parse.add_argument("--device", default="cpu", type=str)
    parse.add_argument("--save_path", default=r"./weight01.pt", type=str)
    parse.add_argument("--video_path", default=r"./1.mp4", type=str)
    parse.add_argument("--result_path", default=r"./result.mp4", type=str)
    args1 = parse.parse_args()
    main(args1)



