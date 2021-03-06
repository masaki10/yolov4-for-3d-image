from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
# from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import csv

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(physical_devices) 
        print(physical_devices[0]) 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    from core.yolov4 import YOLO, decode, compute_loss, decode_train

    trainset = Dataset(FLAGS, is_training=True)
    testset = Dataset(FLAGS, is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    # first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    # second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    epochs = cfg.TRAIN.EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 1])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        raise("not available")
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     model = tf.keras.Model(input_layer, bbox_tensors)
        

    model = tf.keras.Model(input_layer, bbox_tensors)
    # model = tf.keras.Model(input_layer, feature_maps)
    # model.compile(run_eagerly=True)
    # model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        raise
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)


    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    # writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            # if global_steps < warmup_steps:
            #     lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            # else:
            #     lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
            #         (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            #     )
            # optimizer.lr.assign(lr.numpy())

            # writing summary data
            # with writer.as_default():
            #     tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            #     tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            #     tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            #     tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            #     tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            # writer.flush()

            return tf.get_static_value(giou_loss), tf.get_static_value(conf_loss), tf.get_static_value(prob_loss), tf.get_static_value(total_loss)

    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))

            return tf.get_static_value(giou_loss), tf.get_static_value(conf_loss), tf.get_static_value(prob_loss), tf.get_static_value(total_loss)

    best = 1000000.
    f = open("./checkpoint/log.csv", "a", newline="")
    writer = csv.writer(f)
    writer.writerow(["epoch", "giou_loss"])

    for epoch in range(epochs):
        total_giou = 0
        total_conf = 0
        total_prob = 0
        total_loss = 0
        val_total_giou = 0
        val_total_conf = 0
        val_total_prob = 0
        val_total_loss = 0
        cnt = 0
        val_cnt = 0
        print(f"epoch : {epoch}")
        if epoch == 0:
            for name in freeze_layers:
                freeze = model.get_layer(name)
                unfreeze_all(freeze)
        for image_data, target in trainset:
            gi, co, pr, to = train_step(image_data, target)
            total_giou += gi
            total_conf += co
            total_prob += pr
            total_loss += to
            cnt += 1
        
        # cnt = 0
        for image_data, target in testset:
            gi, co, pr, to = test_step(image_data, target)
            val_total_giou += gi
            val_total_conf += co
            val_total_prob += pr
            val_total_loss += to
            val_cnt += 1

        writer.writerow([epoch, total_giou/cnt, total_conf/cnt, total_prob/cnt, total_loss/cnt, val_total_giou/val_cnt, val_total_conf/val_cnt, val_total_prob/val_cnt, val_total_loss/val_cnt])
        if val_total_loss/val_cnt < best:
            best = val_total_loss/val_cnt
            model.save(f"./checkpoint/{epoch}-{val_total_loss/val_cnt}.h5")
        total_giou = 0
        total_conf = 0
        total_prob = 0
        total_loss = 0
        val_total_giou = 0
        val_total_conf = 0
        val_total_prob = 0
        val_total_loss = 0
        cnt = 0
        val_cnt = 0
    
    f.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        print("error")
        pass