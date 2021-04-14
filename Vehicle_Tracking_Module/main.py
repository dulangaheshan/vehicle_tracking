from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import argparse
from os import listdir
from os.path import isfile, join


# def get_model(model_path, config_path, threshold):
#     # Create config
#     cfg = get_cfg()
#     cfg.merge_from_file(config_path)
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
#     cfg.MODEL.WEIGHTS = model_path
#
#     return DefaultPredictor(cfg), cfg
from utils.crop_for_feature_extraction import crop_for_feature_extraction2


def get_model(model_path, threshold):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ('crowdai_vehicle_detection',)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
    cfg.DATASETS.TEST = ('microcontroller/test',)
    # cfg = get_cfg()
    # # cfg.merge_from_file(config_path)
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = 'cpu'

    # # Create config
    # cfg = get_cfg()
    # # cfg.merge_from_file(config_path)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    # cfg.MODEL.WEIGHTS = model_path

    return DefaultPredictor(cfg), cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects from webcam images')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to model')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config')
    parser.add_argument('-t', '--threshold', type=int, default=0.5, help='Detection threshold')
    parser.add_argument('-v', '--video_path', type=str, default='', help='Path to video. If None camera will be used')
    parser.add_argument('-s', '--show', default=True, action="store_false", help='Show output')
    parser.add_argument('-sp', '--save_path', type=str, default='',
                        help='Path to save the output. If None output won\'t be saved')
    args = parser.parse_args()

    # predictor, cfg = get_model(args.model, args.config, args.threshold)
    predictor, cfg = get_model(args.model, args.threshold)

    if args.video_path != '':
        # cap = cv2.VideoCapture(args.video_path) //if isfile(join(img_path, f))
        cameras = [f for f in listdir(args.video_path)]
        for cam in cameras:
            videos = [f for f in listdir(args.video_path + "/" + cam)]
            for video in videos:
                video_path = args.video_path + "/" + cam + "/" + video
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    c1 = 1
                    ret, image = cap.read()

                    outputs = predictor(image)
                    instances = outputs["instances"]
                    scores = instances.get_fields()["scores"].tolist()
                    pred_classes = instances.get_fields()["pred_classes"].tolist()
                    pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()
                    # pred_masks = instances.get_fields()["pred_masks"]
                    print(instances.get_fields())

                    print(c1)
                    crop_for_feature_extraction2(pred_boxes, image, c1, cam)


                    # c1 = c1 + 1
                    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    #
                    # v = Visualizer(image[:, :, ::-1], scale=1.2)
                    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    # if args.show:
                    #     cv2.imshow('object_detection', v.get_image()[:, :, ::-1])
                    #     if cv2.waitKey(25) & 0xFF == ord('q'):
                    #         break

                    # if args.save_path:
                    #     out.write(image)
                cap.release()

    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")

    if args.save_path:
        width = int(cap.get(3))
        height = int(cap.get(4))
        out = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (width, height))

    while cap.isOpened():
        c1 = 1
        ret, image = cap.read()

        outputs = predictor(image)
        instances = outputs["instances"]
        scores = instances.get_fields()["scores"].tolist()
        pred_classes = instances.get_fields()["pred_classes"].tolist()
        pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()
        # pred_masks = instances.get_fields()["pred_masks"]
        print(instances.get_fields())

        print(c1)
        crop_for_feature_extraction2(pred_boxes, image, c1)
        c1 = c1+1
        # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)


        # v = Visualizer(image[:, :, ::-1], scale=1.2)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # if args.show:
        #     cv2.imshow('object_detection', v.get_image()[:, :, ::-1])
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break
        #
        # if args.save_path:
        #     out.write(image)
    cap.release()
    if args.save_path:
        out.release()
    cv2.destroyAllWindows()