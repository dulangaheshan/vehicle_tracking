import argparse
import os
import time
from distutils.util import strtobool
from os import listdir

import cv2

from deep_sort import DeepSort
from detectron2_detection import Detectron2
from util import draw_bboxes
import multiprocessing
import concurrent.futures



class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detectron2 = Detectron2(args)

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)



    def detect_from_detectron(self, im):
        # self.crop_for_feature_extraction(im)
        return self.detectron2.detect(im)

    def deep_sort_detect(self, bbox_xcycwh, cls_conf, im):
        # self.crop_for_feature_extraction(im)
        return self.deepsort.update(bbox_xcycwh, cls_conf, im)



    def detect(self):
        processes = []
        while self.vdo.grab():
            start = time.time()
            _, im = self.vdo.retrieve()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # bbox_xcycwh, cls_conf, cls_ids = self.detectron2.detect(im)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.detect_from_detectron, im)
                return_value = future.result()
                bbox_xcycwh, cls_conf, cls_ids = return_value


            # bbox_xcycwh, cls_conf, cls_ids = self.detect_from_detectron(im)
                if bbox_xcycwh is not None:
                # select class person
                    mask = cls_ids == 0

                    bbox_xcycwh = bbox_xcycwh[mask]
                    bbox_xcycwh[:, 3:] *= 1.2

                    cls_conf = cls_conf[mask]

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self.deep_sort_detect, bbox_xcycwh, cls_conf, im)
                        return_value = future.result()
                        outputs = return_value


                    # outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                        if len(outputs) > 0:
                            bbox_xyxy = outputs[:, :4]
                            identities = outputs[:, -1]
                            im = draw_bboxes(im, bbox_xyxy, identities, args.camera)

                    end = time.time()
                    print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

                    if self.args.display:
                        cv2.imshow("test", im)
                        cv2.waitKey(1)

                    if self.args.save_path:
                        self.output.write(im)
            # exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.3)
    # parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument('-v', '--video_path', type=str, default='', help='Path to video. If None camera will be used')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cameras = [f for f in listdir(args.video_path)]
    for cam in cameras:
        videos = [f for f in listdir(args.video_path + "/" + cam)]
        for video in videos:
            video_path = args.video_path + "/" + cam + "/" + video
            args.VIDEO_PATH = video_path
            args.camera = cam
            args.video = video
            print(video_path)
            with Detector(args) as det:
                det.detect()
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     executor.submit(det.detect())
        # p = multiprocessing.Process(target=det.detect())
        # # processes.append(p)
        # p.start()