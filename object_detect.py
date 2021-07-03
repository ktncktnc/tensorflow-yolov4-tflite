import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from shutil import copyfile
import shutil
import os, glob
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from shutil import copyfile
import shutil
import os, glob
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class ObjectDetector:
    def __init__(self, image_size, output):
        self.model_load = None
        self.type = 'Coco'
        self.image_size = image_size
        self.iou = 0.45
        self.score = 0.25
        self.output = output

    def load_weight(self, weights = './trained/yolov4-tiny-416', is_coco = True):
        if(self.model_load == None or self.type != is_coco):
            print("load weight")
            self.model_load = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
            self.infer = self.model_load.signatures['serving_default']
            self.type = is_coco

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

        cropped_image = utils.draw_bbox(original_image, pred_bbox)

        image = Image.fromarray(cropped_image.astype(np.uint8))
        #if not FLAGS.dont_show:
            #image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        _ , img_name = os.path.split(image_path)

        cv2.imwrite(self.output +  img_name, image)

        result_text = utils.bboxes_to_text(pred_bbox, image.shape)
        return result_text



# #def main():
# def object_detect(image_name, image_size = 416, weights_loaded = './trained/yolov4-tiny-416', framework = 'tf', model="yolov4", tiny=True , iou=0.45, score=0.25, output='./'):
#     imput_image=image_name
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)
#     #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
#     input_size = image_size
#     images = [imput_image]

#     # load model
#     if framework == 'tflite':
#             interpreter = tf.lite.Interpreter(model_path=weights_loaded)
#     else:
#             saved_model_loaded = tf.saved_model.load(weights_loaded, tags=[tag_constants.SERVING])

#     # loop through images in list and run Yolov4 model on each
#     for count, image_path in enumerate(images, 1):
#         original_image = cv2.imread(image_path)
#         original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#         image_data = cv2.resize(original_image, (input_size, input_size))
#         image_data = image_data / 255.

#         images_data = []
#         for i in range(1):
#             images_data.append(image_data)
#         images_data = np.asarray(images_data).astype(np.float32)

#         if framework == 'tflite':
#             interpreter.allocate_tensors()
#             input_details = interpreter.get_input_details()
#             output_details = interpreter.get_output_details()
#             print(input_details)
#             print(output_details)
#             interpreter.set_tensor(input_details[0]['index'], images_data)
#             interpreter.invoke()
#             pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
#             if model == 'yolov3' and tiny == True:
#                 boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
#             else:
#                 boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
#         else:
#             infer = saved_model_loaded.signatures['serving_default']
#             batch_data = tf.constant(images_data)
#             pred_bbox = infer(batch_data)
#             for key, value in pred_bbox.items():
#                 boxes = value[:, :, 0:4]
#                 pred_conf = value[:, :, 4:]

#         boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#             boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#             scores=tf.reshape(
#                 pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#             max_output_size_per_class=50,
#             max_total_size=50,
#             iou_threshold=iou,
#             score_threshold=score
#         )
#         pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
#         #image = utils.draw_bbox(original_image, pred_bbox)
#         cropped_image = utils.draw_bbox(original_image, pred_bbox)
#         # image = utils.draw_bbox(image_data*255, pred_bbox)
#         image = Image.fromarray(cropped_image.astype(np.uint8))
#         #if not FLAGS.dont_show:
#             #image.show()
#         image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

#         _ , img_name = os.path.split(image_name)
#         cv2.imwrite(output +  img_name, image)
#         result_text = utils.bboxes_to_text(pred_bbox, image.shape)
#         return result_text