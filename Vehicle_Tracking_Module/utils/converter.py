import os
import cv2


def converter(videos_dir_path, images_dir_path):
    for root_dir_path, sub_dir_name_list, sub_file_name_list in os.walk(videos_dir_path):
        sub_file_name_list = sorted(sub_file_name_list)
        for seq_idx, sub_file_name in enumerate(sub_file_name_list):
            if not os.path.exists(os.path.join(images_dir_path, os.path.splitext(sub_file_name)[0])):
                os.mkdir(os.path.join(images_dir_path, os.path.splitext(sub_file_name)[0]))
            cap = cv2.VideoCapture(os.path.join(root_dir_path, sub_file_name))
            image_idx = 0
            ret = cap.isOpened()
            while (ret):
                ret, frame = cap.read()
                try:
                    cv2.imwrite(os.path.join(images_dir_path, os.path.splitext(sub_file_name)[0],
                                         "img{}.jpg".format("{}".format(image_idx + 1).zfill(6))), frame)
                    image_idx += 1
                    print(image_idx)
                except Exception as e:
                    print(e)
                    pass
            print("Finish converting {} / {} video seqences into images".format(seq_idx + 1, len(sub_file_name_list)))

