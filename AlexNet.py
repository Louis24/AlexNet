import tensorflow as tf
import time
from pylab import *
from scipy.misc import imread
from numpy import random
from classnames import class_names
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_x = zeros((1, 227, 227, 3)).astype(float32)
train_y = zeros((1, 1000))
dim_x = train_x.shape[1:]
dim_y = train_y.shape[1]

# Read Image RGB->BGR
x_dummy = (random.random((1,) + dim_x) / 255.).astype(float32)
im = x_dummy.copy()
im[0, :, :, :] = (imread('x.png')[:, :, :3]).astype(float32)
im = im - mean(im)
im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]


#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

# Load Weight
net_data = load('bvlc_alexnet.npy', encoding='latin1').item()


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding='VALID', group=1):
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())


x = tf.placeholder(tf.float32, im.shape)

conv1W = tf.Variable(net_data['conv1'][0])
conv1b = tf.Variable(net_data['conv1'][1])
conv1_in = conv(x, conv1W, conv1b, 11, 11, 96, 4, 4, padding='SAME', group=1)
conv1 = tf.nn.relu(conv1_in)

lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

conv2W = tf.Variable(net_data['conv2'][0])
conv2b = tf.Variable(net_data['conv2'][1])
conv2_in = conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding='SAME', group=2)
conv2 = tf.nn.relu(conv2_in)

lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)

maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

conv3W = tf.Variable(net_data['conv3'][0])
conv3b = tf.Variable(net_data['conv3'][1])
conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding='SAME', group=1)
conv3 = tf.nn.relu(conv3_in)

conv4W = tf.Variable(net_data['conv4'][0])
conv4b = tf.Variable(net_data['conv4'][1])
conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding='SAME', group=2)
conv4 = tf.nn.relu(conv4_in)

conv5W = tf.Variable(net_data['conv5'][0])
conv5b = tf.Variable(net_data['conv5'][1])
conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding='SAME', group=2)
conv5 = tf.nn.relu(conv5_in)

maxpool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

fc6W = tf.Variable(net_data['fc6'][0])
fc6b = tf.Variable(net_data['fc6'][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

fc7W = tf.Variable(net_data['fc7'][0])
fc7b = tf.Variable(net_data['fc7'][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

fc8W = tf.Variable(net_data['fc8'][0])
fc8b = tf.Variable(net_data['fc8'][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

prob = tf.nn.softmax(fc8)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

t = time.time()
output = sess.run(prob, feed_dict={x: im})
print('time elapsed:', (time.time() - t))

# Output

index = argsort(output)[0, :]
for i in range(5):
    print(class_names[index[-1 - i]], output[0, index[-1 - i]])
