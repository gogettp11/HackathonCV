from configparser import Interpolation
import numpy as np
import tensorflow as tf
import tensorboard as tb
import os
import cv2 as cv
from datetime import datetime


class Decoder(tf.keras.Model):

    def __init__(self, outputs):
        super().__init__()
        self.reduction1 = tf.keras.layers.Conv2D(320, 1, input_shape=[16, 16, 1280]) # 16x16
        self.upsample1 = tf.keras.layers.UpSampling2D(size=(4, 4))  #64x64
        self.conv1 = tf.keras.layers.Conv2D(576, 33) # 32x32
        self.reduction2 = tf.keras.layers.Conv2D(576, 1)  # 32x32
        self.upsample2 = tf.keras.layers.UpSampling2D(size=(8, 8)) # 256x256
        self.conv2 = tf.keras.layers.Conv2D(576, 65, strides=3) # 32 x 32
        self.reduction3 = tf.keras.layers.Conv2D(4, 1)  # 64x64
        self.upsample3 = tf.keras.layers.UpSampling2D(size=(4, 4)) #256x256
        self.conv3 = tf.keras.layers.Conv2D(4, 129) # 128x128
        self.reduction4 = tf.keras.layers.Conv2D(4, 1) 
        self.upsample4 = tf.keras.layers.UpSampling2D(size=(4, 4)) #512x512
        self.reduction5 = tf.keras.layers.Conv2D(1, 1)

    def call(self, inputs): # 0-256x256x96 1-128x128x144 2-64x64x192 3-32x32x576 4-16x16x320
        x = self.reduction1(inputs)
        # tf.concat([x, residuals[4]],axis=-1)
        x = self.upsample1(x)
        # tf.concat([x, residuals[2]],axis=-1)
        x = self.conv1(x)
        # tf.concat([x, residuals[3]], axis=-1)
        x = self.reduction2(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.reduction3(x)
        x = self.upsample3(x)
        # tf.concat([x, residuals[0]], axis=-1)
        x = self.conv3(x)
        x = self.reduction4(x)
        x = self.upsample4(x)
        x = self.reduction5(x)
        # x = tf.math.sigmoid(x)
        return x

class Custom_model(tf.keras.Model):

    def __init__(self, input_shape=[512, 512, 3], output_shape=[512, 512, 1]) -> None:
        super().__init__()
        self.encoder = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=input_shape, include_top=False)
        self.encoder.trainable = True

        # Use the activations of these layers
        self.layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]


        self.decoder = Decoder(output_shape)

    def call(self, inputs):
        x = self.encoder(inputs)
        # base_model_outputs = [self.encoder.get_layer(name).output for name in self.layer_names]
        # base_model_outputs = [self.encoder.get_layer(name) for name in self.layer_names]
        # base_model_outputs = [tf.convert_to_tensor(np.array(x)) for x in base_model_outputs]
        x = self.decoder(x)
        return x

def focal_loss(alpha=0.25, gamma=2):
  def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    targets = tf.cast(targets, tf.float32)
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

  def loss(y_true, logits):
    y_pred = tf.math.sigmoid(logits)
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)

  return loss

output_source = './Hackathon_Sample/Ground_truth'
input_source = './Hackathon_Sample/Images'

data_count = len([name for name in os.listdir(output_source)])

batch_outputs = tf.constant([cv.imread(
    f"{output_source}/mask_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], shape=(data_count, 512, 512, 1), dtype=tf.float64)

batch_inputs = tf.constant([[cv.imread(
    f"{input_source}/Aspect/aspect_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
        f"{input_source}/DTM/dtm_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
            f"{input_source}/Flow_Accum/flowacc_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                f"{input_source}/Flow_Direction/flowdir_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                    f"{input_source}/Orthophoto/ortho_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                        f"{input_source}/Prof_curv/pcurv_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                            f"{input_source}/Slope/slope_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                                f"{input_source}/Tang_curv/tcurv_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)], [cv.imread(
                                    f"{input_source}/Topo_Wetness/twi_{i}.png", cv.IMREAD_GRAYSCALE) for i in range(100, 100+data_count)]], shape=(data_count, 512, 512, 9))

# transformation
# batch_outputs = tf.math.round((batch_outputs)/128)
batch_outputs = tf.where(batch_outputs < 128, 0, batch_outputs)
batch_outputs = tf.where(batch_outputs == 128, 0.5, batch_outputs)
batch_outputs = tf.where(batch_outputs == 255, 1, batch_outputs)

# build model
model = Custom_model()

# params
data_indicies = [x for x in range(data_count)]
batch_size = 2
log_dir = f"./logs/{datetime.now().strftime('%H%M%S')}"
writer = tf.summary.create_file_writer(log_dir)
writer.set_as_default()
loss_fun = tf.keras.losses.MeanSquaredError()
opt = tf.optimizers.Adam(learning_rate=0.0001)
# train
for step in range(50000):
    # train
    indices = np.random.choice(data_indicies, size=batch_size)

    input = tf.gather(batch_inputs, indices, axis=0, batch_dims=1)
    input = tf.gather(input, [0, 1, 2], axis=3)

    output = tf.gather(batch_outputs, indices, axis=0, batch_dims=1)

    with tf.GradientTape() as tape:
        out = model(input)
        loss = loss_fun(output, out)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    tf.summary.scalar(f'loss train', loss, step=step)
    tf.summary.image(f'segmentation', out, step=step)

    # # test
    # indices = np.random.choice(data_indicies, size=8)
    # batch_input = tf.constant([np.load(
    #     f"{test_source}/images/{i}", allow_pickle=True) for i in indices], dtype=tf.float64)
    # batch_output = tf.constant([output_data_test[i] for i in indices],
    #                            dtype=tf.float64)

    # out, images = model(batch_input)
    # loss = loss_fun(batch_output, out)
    # tf.summary.scalar(f'loss test', loss, step=step)

print("end")
