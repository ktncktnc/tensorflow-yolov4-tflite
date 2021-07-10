import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
from core.config import cfg
import numpy as np
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class ObjectDetector:
    def __init__(self, output):
        self.model_load = None
        self.type = 'Coco'
        self.image_size = 418
        self.iou = 0.45
        self.score = 0.25
        self.output = output

    def load_weight(self, weights = './trained/yolov4-tiny-416', is_coco = True, img_size = 418):
        if(self.model_load == None or self.type != is_coco):
            print("load weight")
            self.model_load = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
            self.infer = self.model_load.signatures['serving_default']
            self.type = is_coco
        self.image_size = img_size    

    def detect(self, image_path):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (self.image_size, self.image_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        
        batch_data = tf.constant(images_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold= self.iou,
            score_threshold= self.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        if self.type == True:
            cropped_image = utils.draw_bbox(original_image, pred_bbox)
        else:
            cropped_image = utils.draw_bbox(image = original_image, bboxes = pred_bbox, classes = utils.read_class_names(cfg.UDACITY.CLASSES))    

        image = Image.fromarray(cropped_image.astype(np.uint8))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        _ , img_name = os.path.split(image_path)

        cv2.imwrite(self.output +  img_name, image)

        if self.type == True:
            result_text = utils.bboxes_to_text(pred_bbox, image.shape)
        else:
            result_text = utils.bboxes_to_text(bboxes = pred_bbox, image_shape = image.shape, classes = utils.read_class_names(cfg.UDACITY.CLASSES))    

        return result_text