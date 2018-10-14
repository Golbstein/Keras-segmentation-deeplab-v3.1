from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output
import numpy as np
from deeplabv3p import Deeplabv3, preprocess_input
from PIL import Image
import os
import matplotlib.pyplot as plt
import multiprocessing
workers = multiprocessing.cpu_count()//2
import keras
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import *
from keras.models import Model, Sequential
#import bcolz
import itertools
import pandas as pd
from keras.callbacks import TensorBoard
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from segnet import *
from clr import *
from keras.regularizers import l2
from keras.utils import to_categorical
from enet import build


# mobilenet
def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_%d_bn' % block_id)(x)
    return Activation(ReLU(6.), name='conv_%d_relu' % block_id)(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(ReLU(6.), name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(ReLU(6.), name='conv_pw_%d_relu' % block_id)(x)

def _resize_images(x, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = K.permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_bilinear(x, new_shape)
        x = K.permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape(
            (None, None, original_shape[2] * height_factor if original_shape[2] is not None else None,
             original_shape[3] * width_factor if original_shape[3] is not None else None))
        return x
    elif data_format == 'channels_last':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        x = tf.image.resize_bilinear(x, new_shape)
        x.set_shape(
            (None, original_shape[1] * height_factor if original_shape[1] is not None else None,
             original_shape[2] * width_factor if original_shape[2] is not None else None, None))
        return x
    else:
        raise ValueError('Invalid data_format:', data_format)

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])
    def call(self, inputs, **kwargs):
        return _resize_images(inputs, self.size[0], self.size[1],
                              self.data_format)

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def MobileUNet(input_shape=None,
               alpha=1.0,
               alpha_up=1.0,
               depth_multiplier=1,
               dropout=1e-3,
               input_tensor=None, n_classes = 22):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    b00 = _conv_block(img_input, 32, alpha, strides=(2, 2), block_id=0)
    b01 = _depthwise_conv_block(b00, 64, alpha, depth_multiplier, block_id=1)

    b02 = _depthwise_conv_block(b01, 128, alpha, depth_multiplier, block_id=2, strides=(2, 2))
    b03 = _depthwise_conv_block(b02, 128, alpha, depth_multiplier, block_id=3)

    b04 = _depthwise_conv_block(b03, 256, alpha, depth_multiplier, block_id=4, strides=(2, 2))
    b05 = _depthwise_conv_block(b04, 256, alpha, depth_multiplier, block_id=5)

    b06 = _depthwise_conv_block(b05, 512, alpha, depth_multiplier, block_id=6, strides=(2, 2))
    b07 = _depthwise_conv_block(b06, 512, alpha, depth_multiplier, block_id=7)
    b08 = _depthwise_conv_block(b07, 512, alpha, depth_multiplier, block_id=8)
    b09 = _depthwise_conv_block(b08, 512, alpha, depth_multiplier, block_id=9)
    b10 = _depthwise_conv_block(b09, 512, alpha, depth_multiplier, block_id=10)
    b11 = _depthwise_conv_block(b10, 512, alpha, depth_multiplier, block_id=11)

    b12 = _depthwise_conv_block(b11, 1024, alpha, depth_multiplier, block_id=12, strides=(2, 2))
    b13 = _depthwise_conv_block(b12, 1024, alpha, depth_multiplier, block_id=13)
    # b13 = Dropout(dropout)(b13)

    filters = int(512 * alpha)
    up1 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b13),
        b11,
    ], axis=3)
    b14 = _depthwise_conv_block(up1, filters, alpha_up, depth_multiplier, block_id=14)

    filters = int(256 * alpha)
    up2 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b14),
        b05,
    ], axis=3)
    b15 = _depthwise_conv_block(up2, filters, alpha_up, depth_multiplier, block_id=15)

    filters = int(128 * alpha)
    up3 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b15),
        b03,
    ], axis=3)
    b16 = _depthwise_conv_block(up3, filters, alpha_up, depth_multiplier, block_id=16)

    filters = int(64 * alpha)
    up4 = concatenate([
        Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(b16),
        b01,
    ], axis=3)
    b17 = _depthwise_conv_block(up4, filters, alpha_up, depth_multiplier, block_id=17)

    filters = int(32 * alpha)
    up5 = concatenate([b17, b00], axis=3)
    # b18 = _depthwise_conv_block(up5, filters, alpha_up, depth_multiplier, block_id=18)
    b18 = _conv_block(up5, filters, alpha_up, block_id=18)

    x = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', activation='linear')(b18)
    x = BilinearUpSampling2D(size=(2, 2))(x)
    
    x = Reshape((-1, n_classes))(x)
    
    x = Activation('softmax')(x)

    model = Model(img_input, x)
    return model

# Tiramisu
def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization()(x)
def relu_bn(x): return relu(bn(x))
def concat(xs): return Concatenate()(xs)
def conv(x, nf, sz, wd, p, stride=1): 
    x = Conv2D(nf, sz, padding='same', strides=(stride, stride))(x)
    return dropout(x, p)
def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1): 
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)
def dense_block(n,x,growth_rate,p,wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x, added
def transition_dn(x, p, wd):
#     x = conv_relu_bn(x, x.shape.as_list()[-1], sz=1, p=p, wd=wd)
#     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, 
                        x.shape.as_list()[-1], sz=1, 
                        p=p, wd=wd, stride=2)
def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added

def transition_up(added, wd=0):
    x = concat(added)
    _,r,c,ch = x.shape.as_list()
    return Conv2DTranspose(ch, 3, padding='same', 
                           strides=(2,2))(x)
def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i,n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x,skips[i]])
        x, added = dense_block(n,x,growth_rate,p,wd)
    return x
def reverse(a): return list(reversed(a))
def create_tiramisu(nb_classes, img_input, nb_dense_block=6, 
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)
    x = conv(x, nb_classes, 1, wd, 0)
    _,r,c,f = x.shape.as_list()
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)
def Tiramisu(input_shape, n):
    img_input = Input(shape=input_shape)
    x = create_tiramisu(n, img_input, 
                        nb_layers_per_block=[4,5,7,10,12,15], p=0.2, wd=1e-4)
    model = Model(img_input, x)
    
    return model

    
# output folder and model names


from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib


def convert_keras_to_pb(model, models_dir, model_filename):
    from tensorflow.python.framework import graph_io
    from tensorflow.python.tools import freeze_graph
    from tensorflow.core.protobuf import saver_pb2
    from tensorflow.python.training import saver as saver_lib
    output_node = [node.op.name for node in model.outputs][0]
    K.set_learning_phase(0)
    sess = K.get_session()
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
    checkpoint_path = saver.save(sess, './saved_ckpt', global_step=0, latest_filename='checkpoint_state')
    graph_io.write_graph(sess.graph, '.', 'tmp.pb')
    froze = freeze_graph.freeze_graph('./tmp.pb', '',
                                      False, checkpoint_path, output_node,
                                      "save/restore_all", "save/Const:0",
                                      models_dir+model_filename, False, "")
    # output folder and model names
    #models_dir = './models/'
    #model_filename = 'model_tf_{}x{}.pb'.format(image_size[0], image_size[1])
    #convert_keras_to_pb(deeplab_model, models_dir, model_filename)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        #print('freeze', freeze_var_names)
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        
        #frozen_graph = freeze_session(K.get_session(),
        #                      output_names=[out.op.name for out in deeplab_model.outputs])
        #tf.train.write_graph(frozen_graph, "weights", "arch1.pb", as_text=False)

        return frozen_graph


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def create_unet(image_size, n_classes):
    s = Input(image_size+(3,))
    c1 = Conv2D(16, 3, activation='relu', padding='same') (s)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(16, 3, activation='relu', padding='same') (c1)
    p1 = MaxPooling2D() (c1)
    c2 = Conv2D(32, 3, activation='relu', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32, 3, activation='relu', padding='same') (c2)
    p2 = MaxPooling2D() (c2)
    c3 = Conv2D(64, 3, activation='relu', padding='same') (p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same') (c3)
    p3 = MaxPooling2D() (c3)
    c4 = Conv2D(128, 3, activation='relu', padding='same') (p3)
    c4 = Conv2D(128, 3, activation='relu', padding='same') (c4)
    p4 = MaxPooling2D() (c4)
    c5 = Conv2D(256, 3, activation='relu', padding='same') (p4)
    c5 = Conv2D(256, 3, activation='relu', padding='same') (c5)
    u6 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same') (c5)
    u6 = Concatenate(axis=3)([u6, c4])
    c6 = Conv2D(128, 3, activation='relu', padding='same') (u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same') (c6)
    u7 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same') (c6)
    u7 = Concatenate(axis=3)([u7, c3])
    c7 = Conv2D(64, 3, activation='relu', padding='same') (u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same') (c7)
    u8 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same') (c7)
    u8 = Concatenate(axis=3)([u8, c2])
    c8 = Conv2D(32, 3, activation='relu', padding='same') (u8)
    c8 = Conv2D(32, 3, activation='relu', padding='same') (c8)
    u9 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same') (c8)
    u9 = Concatenate(axis=3)([u9, c1])
    c9 = Conv2D(16, 3, activation='relu', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(16, 3, activation='relu', padding='same') (c9)
    c9 = BatchNormalization()(c9)
    outputs = Conv2D(n_classes, 1) (c9)
    outputs = Reshape((image_size[0]*image_size[1], n_classes)) (outputs)
    outputs = Activation('softmax') (outputs)
    
    model = Model(inputs=[s], outputs=[outputs])
    return model
    
def load_test_data(PATH, size, normalize_image = True):
    images = os.listdir(PATH + 'JPEGImages/test')
    test = []
    for j in images:
        im = Image.open(PATH+'JPEGImages/test/'+j)
        test.append(np.array(im.resize(size)))
    test = np.array(test).astype('float32')
    if normalize_image:
        test = preprocess_input(test)
    return test
    
def load_train_data(PATH, size, normalize_image = True):
    images = os.listdir(PATH + 'JPEGImages/train')
    segClass = os.listdir(PATH + 'SegmentationClassAug')
    x = []
    y = []
    for j in images:
        try:
            seg = Image.open(PATH + 'SegmentationClassAug/' + j[:-3]+'png')
            y.append(np.array(seg.resize(size)))
            im = Image.open(PATH+'JPEGImages/train/'+j)
            x.append(np.array(im.resize(size)))
        except:
            os.rename(PATH+'JPEGImages/train/'+j, PATH+'JPEGImages/test/'+j)
    y = np.array(y)
    x = np.array(x)
    x = x.astype('float32')
    if normalize_image:
        x = preprocess_input(x)
    y = np.expand_dims(y, 3)
    y = y.astype('int64')
    return x, y
    
    
def evaluate_model(preds, labels, data_from = 'test'):
    conf_m, IOU, meanIOU, mean_acc = calculate_iou(preds, labels)
    classes = [c for c in get_VOC2012_classes().values()]
    classes.append('average')
    ious = np.append(IOU, meanIOU)
    d = {'Objects': classes, 'IoU '+ data_from: ious}
    df = pd.DataFrame.from_dict(d)
    return df, conf_m, mean_acc

#def save_array(fname, arr):
#    c=bcolz.carray(arr, rootdir=fname, mode='w')
#    c.flush()

#def load_array(fname):
#    return bcolz.open(fname)[:]

def get_VOC2012_classes():
    PASCAL_VOC_classes = {
        0: 'background', 
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'dining table',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'potted plant',
        17: 'sheep',
        18: 'sofa',
        19 : 'train',
        20 : 'tv',
        21 : 'void'
    }
    return PASCAL_VOC_classes

weights = {0: .05, 1: .1, 2: 1., 3: .5, 4: 1., 5: 1., 6: 1., 7: 1., 8: 1., 9: 1.,
           10: .8, 11: .6, 12: 1.,13: .7, 14: 1., 15: 1.2, 16: 1., 17: .6, 18: .9, 19: 1., 20: 1., 255: 0.}

def label2weight(y):
    return weights[y]

def get_class_weights(b = .1):
    n_classes = len(get_VOC2012_classes())
    class_weights = {i : 1. for i in range(n_classes)}
    class_weights[0] = b
    class_weights[n_classes-1] = 0
    return class_weights
    
def segmentation_generator(image_gen, mask_gen):
    while True:
            X = image_gen.next()
            y = mask_gen.next()
            y = np.reshape(y, (-1, np.prod(mask_gen.image_shape), 1))
            y = y.astype('int32')
            
            org_shape = y.shape
            res = map(label2weight, y.flatten())
            sample_weights = np.reshape(np.array(list(res)), org_shape[:2])
            
#             sample_weights = np.ones(y.shape)
#             sample_weights[y==0] = .1
#             sample_weights[y==255] = 0

            y[y==255]=21
            yield X, y, sample_weights

def preprocess_mask(x):
    x = x.astype('int32')
    th = round(x.shape[0]*x.shape[1]*0.005) # object at size less then 1% of the whole image
    x[(x>20) & (x<255)] = 255
    ctr = Counter(x.flatten())
    for k in ctr.keys():
        if ctr[k]<th:
            x[x==k] = 255
#     x[x>20] = 21
    return x

def foreground_sparse_accuracy(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.to_int32(K.reshape(y_true, (-1, 1))[:,0])
    y_true = K.one_hot(y_true, nb_classes+1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = tf.cast(unpacked[-1], tf.bool) | tf.cast(unpacked[0], tf.bool)
    true_pixels = K.argmax(y_true, axis=-1) # exclude background
    pred_pixels = K.argmax(y_pred, axis=-1)
    return K.sum(tf.to_float(~legal_labels & K.equal(true_pixels, pred_pixels))) / K.sum(tf.to_float(legal_labels))

def background_sparse_accuracy(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    pred_pixels = K.argmax(y_pred, axis=-1)
    true_pixels = tf.to_int64(K.reshape(y_true, (-1, 1))[:,0])
    legal_labels = K.equal(true_pixels, 0)
    return K.sum(tf.to_float(legal_labels & K.equal(true_pixels, pred_pixels))) / K.sum(tf.to_float(legal_labels))

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)
    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1),
                                                    K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = tf.to_int64(y_true[:,:,0])
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(true_pixels, nb_classes)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)


def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)
    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)
    return cross_entropy_mean

# def wisense_loss(w):
#     def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
#         y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
#         log_softmax = tf.nn.log_softmax(y_pred)
#         y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
#         unpacked = tf.unstack(y_true, axis=-1)
#         y_true = tf.stack(unpacked[:-1], axis=-1)
#         cross_entropy_bg = -(y_true[:,0] * log_softmax[:,0])
#         cross_entropy_fg = -K.sum(y_true[:,1:] * log_softmax[:,1:], axis=1)
#         cross_entropy_mean = K.mean(w * cross_entropy_fg + cross_entropy_bg)
#         return cross_entropy_mean
#     return softmax_sparse_crossentropy_ignoring_last_label

def show_aug_data(gen, data_trn_gen_args_image, data_trn_gen_args_mask):
    x, y = next(gen)
    x+=1
    x/=2
    x = x[0]
    y = y[0]
    x = x.reshape((1,) + x.shape) 
    y = y.reshape((1,) + y.shape) 

    gen_mask = ImageDataGenerator(**data_trn_gen_args_mask)
    data_trn_gen_args_image['preprocessing_function'] = None
    gen = ImageDataGenerator(**data_trn_gen_args_image)

    mask_itr = gen_mask.flow(y, batch_size=1, seed=7)
    trn_itr = gen.flow(x, batch_size=1, seed=7)
    itr = zip(trn_itr, mask_itr)

    plt.figure(figsize=(15,7))
    i = 0
    while(True):
        i += 1
        if i > 8:
            break
        batch, y = next(itr)
        plt.subplot(2,4,i)
        plt.imshow(batch[0])
        y[y==255] = 0
        plt.imshow(y[0,:,:,0], alpha=.5)
    classes = get_VOC2012_classes()
    print([classes[j] for j in np.unique(y).astype('int')])
    
def calculate_iou(y_preds, labels):
    nb_classes = y_preds.shape[-1]
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    total = len(labels)
    mean_acc = 0.
    preds = np.argmax(y_preds, axis = -1)
    for i in range(total):
        flat_pred = np.ravel(preds[i])
        flat_label = np.ravel(labels[i])
        acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == (nb_classes-1):
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', i)
            if l==p:
                acc+=1
        acc /= flat_pred.shape[0]
        mean_acc += acc
    mean_acc /= total
    print('mean acc: %f'%mean_acc)
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU, mean_acc

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    #plt.figure(figsize=(18,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90,fontsize=9)
    plt.yticks(tick_marks, classes,fontsize=9)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j],2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    #plt.tight_layout()
    plt.ylabel('True label',fontsize=9)
    plt.xlabel('Predicted label',fontsize=9)
    plt.colorbar()

class SegModel:
    
    epochs = 20
    batch_size = 16
    def __init__(self, dataset='VOCdevkit/VOC2012', image_size=(320,320), use_coords = True):
        self.coords = use_coords
        self.sz = image_size
        self.mainpath = dataset
        self.n_val = 0
        
    def build_callbacks(self, tf_board = False, plot_process = True, steps = 50):
        
        tensorboard = TensorBoard(log_dir='./logs/'+self.net, histogram_freq=0,
                          write_graph=False, write_images = False)

        cl = CyclicLR(base_lr=0.0001, max_lr=0.01,
                      step_size=steps, mode = 'cosine', gamma = 0.999995,
                      scale_mode='iterations', cycle_mult = 2)
        
        checkpointer = ModelCheckpoint(filepath = self.modelpath, verbose=1, save_best_only=True, save_weights_only=True)
        stop_train = EarlyStopping(monitor = 'val_loss', patience=10, verbose=1)
        #reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5,
        #            patience=4, min_lr=0.00001)
        if plot_process:
            callbacks = [checkpointer, cl, stop_train, tensorboard, PlotLearning()]
        elif tf_board:
            callbacks = [checkpointer, cl, stop_train, tensorboard]
        else:
            callbacks = [checkpointer, cl, stop_train]
        return callbacks, cl
    
    
    def create_seg_model(self, opt, net, load_weights = False, multi_gpu = True, to_compile = True):
        
        self.net = net
        n_classes = len(get_VOC2012_classes()) - 1
        
        if net == 'enet':
            self.modelpath = 'weights/enet.h5'
            model, _ = build(nc=n_classes, w=self.sz[1], h=self.sz[0], plot=False)
        elif net == 'mobileunet':
            self.modelpath = 'weights/mobileunet.h5'
            model = MobileUNet(input_shape=(self.sz[1], self.sz[0], 3),
                               alpha=1, alpha_up=1, depth_multiplier=1, n_classes = n_classes)
        elif net == 'unet':
            self.modelpath = 'weights/unet.h5'
            model = create_unet(self.sz, n_classes)
        elif net == 'segnet':
            self.modelpath = 'weights/segnet.h5'
            model = create_segnet(self.sz, n_classes)
        elif net == 'tiramisu':
            self.modelpath = 'weights/tiramisu.h5'
            model = Tiramisu(self.sz+(3,), n_classes)
        else:
            self.modelpath = 'weights/deeplabv3.h5'
            model = Deeplabv3(weights=None, input_tensor=None, 
                              input_shape = self.sz + (3,), classes=21, 
                              backbone='mobilenetv2', OS=16, alpha=1, use_coordconv = self.coords)
        if self.coords:
            self.modelpath = 'weights/deeplabv3_coordconv.h5'

        if multi_gpu:
            from keras.utils import multi_gpu_model
            model = multi_gpu_model(model, gpus = len(get_available_gpus()))
            
        if to_compile:
            model.compile(optimizer = opt, sample_weight_mode = "temporal",
                          loss = softmax_sparse_crossentropy_ignoring_last_label,
                          #loss = wisense_loss(30),
                          metrics = [Mean_IOU, background_sparse_accuracy, 
                                     sparse_accuracy_ignoring_last_label,
                                     foreground_sparse_accuracy, 'acc'])
        if load_weights:
            self.load_weights(model)
        return model
    
    def create_generators(self, data_gen_args_image, data_gen_args_mask, subset = 'training'):
    
        mask_datagen = ImageDataGenerator(**data_gen_args_mask)
        image_datagen = ImageDataGenerator(**data_gen_args_image)
    
        if subset=='training':
            shuffle = True
        else:
            shuffle = False
        
        image_generator = image_datagen.flow_from_directory(self.mainpath+'JPEGImages', 
                                                            target_size=self.sz,
                                                            classes = ['train'], 
                                                            class_mode=None, 
                                                            batch_size=self.batch_size, shuffle=shuffle, 
                                                            seed=29, subset = subset)

        mask_generator = mask_datagen.flow_from_directory(self.mainpath, target_size=self.sz, color_mode = 'grayscale',
                                                          classes = ['SegmentationClassAug'], 
                                                          class_mode=None, 
                                                          batch_size=self.batch_size, 
                                                          shuffle=shuffle, 
                                                          seed=29, subset = subset)
        if subset == 'training':
            self.n_trn = image_generator.n
            self.step = self.n_trn/self.batch_size
        else:
            self.n_val = image_generator.n
        
        return segmentation_generator(image_generator, mask_generator)
        #return zip(image_generator, mask_generator)

    def load_weights(self, model):
        model.load_weights(self.modelpath)
        
    def train_generator(self, model, train_generator, valid_generator, tf_board = True, mp = True):
        steps = self.n_trn//self.batch_size
        callbacks, cl = self.build_callbacks(tf_board = tf_board, plot_process = False, steps=steps//2)
        h = model.fit_generator(train_generator, class_weight = None,
                                steps_per_epoch=steps, 
                                epochs = self.epochs, verbose=1, 
                                callbacks = callbacks, 
                                validation_data=valid_generator, 
                                validation_steps=self.n_val//self.batch_size, 
                                max_queue_size=10, 
                                workers=workers, use_multiprocessing=mp)
        return h, cl
    
    def train(self, model, X, y, val_data, tf_board = False, plot_train_process = True):
        h = model.fit(X, y, validation_data = val_data, verbose=1, 
                      batch_size = self.batch_size, epochs = self.epochs, 
                      callbacks = self.build_callbacks(tf_board = tf_board, plot_process = plot_train_process))
        return h
    
    @classmethod
    def set_num_epochs(cls, new_epochs):
        cls.epochs = new_epochs
    @classmethod
    def set_batch_size(cls, new_batch_size):
        cls.batch_size = new_batch_size

# inheritance for training process plot 
# class PlotLearning(keras.callbacks.Callback, SegModel):
#     def __init__(self):
#         SegModel.__init__(self)
#     def on_train_begin(self, logs={}):
#         self.i = 0
#         self.x = []
#         self.losses = []
#         self.val_losses = []
#         self.acc = []
#         self.val_acc = []
#         self.acc2 = []
#         self.val_acc2 = []

# #        self.fig = plt.figure()
#         self.logs = []
#     def on_epoch_end(self, epoch, logs={}):
#         self.logs.append(logs)
#         self.x.append(self.i)
#         self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.acc.append(logs.get('foreground_sparse_accuracy'))
#         self.val_acc.append(logs.get('val_foreground_sparse_accuracy'))
#         self.acc2.append(logs.get('background_sparse_accuracy'))
#         self.val_acc2.append(logs.get('val_background_sparse_accuracy'))

#         self.i += 1
#         fig, ax1 = plt.subplots(figsize=(10,5));
#         clear_output(wait=True)
#         lns1 = ax1.plot(self.x, self.acc, label="Sparse accuracy foreground", color=[0,0,1], linewidth = 3.);
#         lns2 = ax1.plot(self.x, self.val_acc, label="Sparse validation accuracy foreground", color=[0,0,0.5], linewidth = 3.);
        
#         lns11 = ax1.plot(self.x, self.acc2, label="Sparse accuracy background", color=[0,1,0], linewidth = 3.);
#         lns21 = ax1.plot(self.x, self.val_acc2, label="Sparse validation accuracy background", color=[0,0.5,0], linewidth = 3.);

#         ax1.set_ylabel('Sparse accuracy',fontsize=15);
#         ax1.set_xlabel('Epoch',fontsize=15);
#         ax1.set_title('Deeplab V3+ Segmentation Model');
#         ax2 = ax1.twinx()
#         lns3 = ax2.plot(self.x, self.losses, '--', label="loss", color=[1,0,0]);
#         lns4 = ax2.plot(self.x, self.val_losses, '--', label="validation loss", color=[0.5,0,0]);
#         ax2.set_ylabel('Loss',fontsize=15)
#         lns = lns1+lns2+lns3+lns4+lns11+lns21
#         ax1.legend(lns, [l.get_label() for l in lns])
#         ax1.grid(True)
#         plt.xlim([-0.05, self.epochs+.05])
#         plt.show();

#class PlotLearning(keras.callbacks.Callback, SegModel):
#    def __init__(self):
#        SegModel.__init__(self)
#    def on_train_begin(self, logs={}):
#        self.i = 0
#        self.x = []
#        self.losses = []
#        self.val_losses = []
#        self.acc = []
#        self.val_acc = []
#        self.acc2 = []
#        self.val_acc2 = []
#        self.fig = plt.figure()
#        self.logs = []
#    def on_epoch_end(self, epoch, logs={}):
#        self.logs.append(logs)
#        self.x.append(self.i)
##        self.losses.append(logs.get('loss'))
#       self.val_losses.append(logs.get('val_loss'))
#        self.acc.append(logs.get('foreground_sparse_accuracy'))
#        self.val_acc.append(logs.get('val_foreground_sparse_accuracy'))
#        self.i += 1
#        fig, ax1 = plt.subplots(figsize=(10,5));
#       clear_output(wait=True)
#        lns1 = ax1.plot(self.x, self.acc, label="Sparse accuracy foregroung", color='blue', linewidth = 3.);
#        lns2 = ax1.plot(self.x, self.val_acc, label="Sparse validation accuracy foregroung", color='green', linewidth = 3.);
###        
  #      lns11 = ax1.plot(self.x, self.acc2, label="Sparse accuracy background", color='black', linewidth = 3.);
  #      lns21 = ax1.plot(self.x, self.val_acc2, label="Sparse validation accuracy background", color='m', linewidth = 3.);

       # ax1.set_ylabel('Sparse accuracy',fontsize=15);
       # ax1.set_xlabel('Epoch',fontsize=15);
       # ax2 = ax1.twinx()
      #  lns3 = ax2.plot(self.x, self.losses, '--', label="loss", color='orange');
      #  lns4 = ax2.plot(self.x, self.val_losses, '--', label="validation loss", color='red');
      #  ax2.set_ylabel('Loss',fontsize=15)
      #  lns = lns1+lns2+lns3+lns4+lns11+lns21
      #  ax1.legend(lns, [l.get_label() for l in lns])
      #  ax1.grid(True)
      #  plt.xlim([-0.05, self.epochs+.05])
      #  plt.show();
