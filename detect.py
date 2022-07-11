import tensorflow as tf
from core import utils
from core.yolov4 import YOLO, decode_train, filter_boxes
import numpy as np

CONFIG = {
    "input_size": 160,
    "iou_loss_thresh": 0.5,
    "model_path": "./checkpoint/.h5",
    "image_path": ".vti",
    "score_thres": 0.25,
    "iou_thres": 0.5,
}

if __name__ == "__main__":
    print("hello")
    input_layer = tf.keras.layers.Input([CONFIG["input_size"], CONFIG["input_size"], CONFIG["input_size"], 1])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    IOU_LOSS_THRESH = CONFIG["iou_loss_thresh"]

    freeze_layers = utils.load_freeze_layer("yolov4", False)

    feature_maps = YOLO(input_layer, NUM_CLASS, "yolov4", False)
    
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
            bbox_tensor = decode_train(fm, CONFIG["input_size"] // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        elif i == 1:
            bbox_tensor = decode_train(fm, CONFIG["input_size"] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        else:
            bbox_tensor = decode_train(fm, CONFIG["input_size"] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
        bbox_tensors.append(fm)
        bbox_tensors.append(bbox_tensor)

    
    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights(CONFIG["model_path"])

    img = utils.vtk_data_loader(CONFIG["image_path"])
    img = img.astype("float32") / 255
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2], 1))    
    img = np.array([img])

    pred_box = model(img)

    preds = [pred_box[1], pred_box[3], pred_box[5]]

    bbox_tensors = []
    prob_tensors = []
    for pred in preds:
        pred_prob = pred[:,:,:,:,:,6:7] * pred[:,:,:,:,:,7:]
        pred_prob = tf.reshape(pred_prob, (pred.shape[0], -1, NUM_CLASS))
        pred_xywh = tf.reshape(pred[:,:,:,:,:,:6], (pred.shape[0], -1, 6))
        bbox_tensors.append(pred_xywh)
        prob_tensors.append(pred_prob)
    
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=CONFIG["score_thres"], input_shape=tf.constant([CONFIG["input_size"],CONFIG["input_size"],CONFIG["input_size"]]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
    boxes = pred[:, :, 0:6]
    pred_conf = pred[:, :, 6:]

    nms_input = []
    for idx in range(boxes.shape[1]):
        max_score = np.amax(pred_conf[0][idx])
        cls = np.argmax(pred_conf[0][idx])
        nms_input.append([boxes[0][idx][0], boxes[0][idx][1], boxes[0][idx][2], boxes[0][idx][3], boxes[0][idx][4], boxes[0][idx][5], max_score, cls])
    nms_input = np.array(nms_input)
    result = utils.nms(nms_input, CONFIG["iou_thres"])

    print(result)
    
    ### draw_boxs

    ### crop_boxs
