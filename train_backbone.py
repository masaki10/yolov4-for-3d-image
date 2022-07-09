from pkg_resources import yield_lines
from core import backbone, utils
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint
from scipy.ndimage import zoom
import csv
import random
import numpy as np
import os

CONFIG = {
    "input_size": 32,
    "num_classes": 4,
    "lr": 0.001,
    "train_data_csv_path": "./dataset/train.csv",
    "val_data_csv_path": "./dataset/val.csv",
    "test_data_csv_path": "./dataset/test.csv",
    "batch": 2,
    "epoch": 100,
    "log_path": "./checkpoint/log.csv",
    "checkpoint_dir": "./checkpoint",
}

class Generator:
    def __init__(self, use):
        self.data_dic = {}

        if use == "train":
            with open(CONFIG["train_data_csv_path"]) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.data_dic[row[0]] = int(row[1])

        elif use == "val":
            with open(CONFIG["val_data_csv_path"]) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.data_dic[row[0]] = int(row[1])
        elif use == "test":
            with open(CONFIG["test_data_csv_path"]) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.data_dic[row[0]] = int(row[1])
        else:
            raise

        self.steps = len(self.data_dic) // CONFIG["batch"]

        self.data_lists = list(self.data_dic.keys())

    def __call__(self):
        while True:
            random.shuffle(self.data_lists)
            for step in range(self.steps):
                start_idx = step * CONFIG["batch"]
                end_idx = (step + 1) * CONFIG["batch"]

                inputs = []
                labels = []

                for i in range(start_idx, end_idx):
                    img = utils.vtk_data_loader(self.data_lists[i])
                    img = zoom(img, (CONFIG["input_size"]/img.shape[0], CONFIG["input_size"]/img.shape[0], CONFIG["input_size"]/img.shape[0]))
                    img = img.reshape((img.shape[0], img.shape[1], img.shape[2], 1))
                    img = img.astype("float32") / 255
                    inputs.append(img)
                    # print(img.shape)
                    labels.append(self.data_dic[self.data_lists[i]])
                
                inputs = np.array(inputs)
                # print(labels)
                labels = tf.keras.utils.to_categorical(labels, CONFIG["num_classes"])
                # print(labels)

                yield inputs, labels

def build_model():
    input_layer = tf.keras.layers.Input([CONFIG["input_size"], CONFIG["input_size"], CONFIG["input_size"], 1])
    _, _, output_data = backbone.cspdarknet53(input_layer)
    output_data = tf.keras.layers.GlobalAveragePooling3D()(output_data)
    output_data = tf.keras.layers.Dense(CONFIG["num_classes"])(output_data)
    model = tf.keras.Model(input_layer, output_data)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG["lr"]),
              loss=tf.nn.softmax_cross_entropy_with_logits,
              metrics=['accuracy'])

    model.summary()
    return model


def train():
    train_generator = Generator("train")
    val_generator = Generator("val")

    model = build_model()

    csv_logger = CSVLogger(CONFIG["log_path"], append=True, separator=",")

    model_checkpoint = ModelCheckpoint(os.path.join(CONFIG["checkpoint_dir"], "weights.{epoch:03d}-{val_loss:.3f}.h5"),
                                                monitor="val_loss", verbose=0, save_best_only=True,
                                                save_weights_only=False, mode='auto', period=1)

    history = model.fit(train_generator(), steps_per_epoch=train_generator.steps,
                                epochs=CONFIG["epoch"], verbose=1, validation_data=val_generator(),
                                validation_steps=val_generator.steps,
                                shuffle=True,
                                callbacks=[csv_logger, model_checkpoint])


if __name__ == "__main__":
    print("hello")
    tf.debugging.experimental.enable_dump_debug_info(
    dump_root="./tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
    train()
    