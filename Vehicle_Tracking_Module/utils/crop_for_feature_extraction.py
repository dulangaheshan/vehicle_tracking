import datetime

import cv2
import os
def crop_for_feature_extraction2(pred_boxes, image, c1, cam):
    print(c1)
    c = 0
    d = str(c) + str(c1) + str(datetime.datetime.now().time())
    for cords in pred_boxes:
        x = int(cords[0])
        y = int(cords[1])
        w = int(cords[2])
        h = int(cords[3])
        print( (x, y), (w, h))
        # crop that part of image which contains desired object
        print(type(image))
        print(image.shape)
        # image = cv2.imencode('.png', image)

        img = image[y:y + h, x:x + w]
        # cv2.imshow("Image", img)
        if not os.path.exists('cropped' + "/" + cam):
            os.mkdir('cropped' + "/" + cam)
        # path = 'cropped'
        c +=1
        cv2.imwrite(os.path.join('cropped' + "/" + cam, 'PImage'+str(d)+'.jpg'), img)

        #
        cv2.waitKey(0)
    # print(cords)