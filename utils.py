from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
from deeplabv3p import Deeplabv3, preprocess_input
from PIL import Image
import os
import keras
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization
from keras.models import Model
#import bcolz
import itertools
import pandas as pd
from keras.callbacks import TensorBoard
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib


def convert_keras_to_pb(model, models_dir, model_filename):
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

# output folder and model names
models_dir = './models/'
model_filename = 'model_tf_{}x{}.pb'.format(image_size[0], image_size[1])

convert_keras_to_pb(deeplab_model, models_dir, model_filename)

tensorboard = TensorBoard(log_dir='./logs1', histogram_freq=2,
                          write_graph=True, write_images=True)



def create_unet(image_size, n_classes):
    s = Input(image_size+(3,))
    c1 = Conv2D(8, 3, activation='relu', padding='same') (s)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(8, 3, activation='relu', padding='same') (c1)
    p1 = MaxPooling2D() (c1)
    c2 = Conv2D(16, 3, activation='relu', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(16, 3, activation='relu', padding='same') (c2)
    p2 = MaxPooling2D() (c2)
    c3 = Conv2D(32, 3, activation='relu', padding='same') (p2)
    c3 = Conv2D(32, 3, activation='relu', padding='same') (c3)
    p3 = MaxPooling2D() (c3)
    c4 = Conv2D(64, 3, activation='relu', padding='same') (p3)
    c4 = Conv2D(64, 3, activation='relu', padding='same') (c4)
    p4 = MaxPooling2D() (c4)
    c5 = Conv2D(128, 3, activation='relu', padding='same') (p4)
    c5 = Conv2D(128, 3, activation='relu', padding='same') (c5)
    u6 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same') (c5)
    u6 = Concatenate(axis=3)([u6, c4])
    c6 = Conv2D(64, 3, activation='relu', padding='same') (u6)
    c6 = Conv2D(64, 3, activation='relu', padding='same') (c6)
    u7 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same') (c6)
    u7 = Concatenate(axis=3)([u7, c3])
    c7 = Conv2D(32, 3, activation='relu', padding='same') (u7)
    c7 = Conv2D(32, 3, activation='relu', padding='same') (c7)
    u8 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same') (c7)
    u8 = Concatenate(axis=3)([u8, c2])
    c8 = Conv2D(16, 3, activation='relu', padding='same') (u8)
    c8 = Conv2D(16, 3, activation='relu', padding='same') (c8)
    u9 = Conv2DTranspose(8, 2, strides=(2, 2), padding='same') (c8)
    u9 = Concatenate(axis=3)([u9, c1])
    c9 = Conv2D(8, 3, activation='relu', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(8, 3, activation='relu', padding='same') (c9)
    c9 = BatchNormalization()(c9)
    outputs = Conv2D(n_classes, 1, activation='sigmoid') (c9)
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
        20 : 'tv'}
    return PASCAL_VOC_classes

def preprocess_mask(x):
    
    x = x.astype('int16')
    th = round(x.shape[0]*x.shape[1]*0.005) # object at size less then 1% of the whole image
    n_classes = len(get_VOC2012_classes())-1
    x[(x>20) & (x<255)] = 255
    ctr = Counter(x.flatten())
    for k in ctr.keys():
        if ctr[k]<th:
            x[x==k] = 255
    return x
                
def foreground_sparse_accuracy(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1) # exclude don't cares
    true_pixels = K.argmax(y_true[:,1:], axis=-1) # exclude background
    pred_pixels = K.argmax(y_pred[:,1:], axis=-1)
    return K.sum(tf.to_float(legal_labels & K.equal(true_pixels, pred_pixels))) / K.sum(tf.to_float(legal_labels))

def background_sparse_accuracy(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    legal_labels = K.equal(true_pixels,0)
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


#def penalized_loss(noise):
#    def loss(y_true, y_pred):
#        return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
#    return loss
def wisense_loss(w):
    def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        log_softmax = tf.nn.log_softmax(y_pred)
        y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)
        cross_entropy_bg = -(y_true[:,0] * log_softmax[:,0])
        cross_entropy_fg = -K.sum(y_true[:,1:] * log_softmax[:,1:], axis=1)
        cross_entropy_mean = K.mean(w * cross_entropy_fg + cross_entropy_bg)
        return cross_entropy_mean
    return softmax_sparse_crossentropy_ignoring_last_label

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
            if l == 255:
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
        
        
    def build_callbacks(self, tf_board = False, plot_process = True):
        checkpointer = ModelCheckpoint(filepath = self.modelpath, verbose=0, save_best_only=True, save_weights_only=True)
        stop_train = EarlyStopping(monitor = 'val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5,
                    patience=4, min_lr=0.00001)
        if tf_board and plot_process:
            callbacks = [checkpointer, reduce_lr, stop_train, tensorboard, PlotLearning()]
        elif plot_process:
            callbacks = [checkpointer, reduce_lr, stop_train, PlotLearning()]
        else:
            callbacks = [checkpointer, reduce_lr, stop_train]
        return callbacks
    
    
    def create_seg_model(self, opt, net, load_weights = False, w=30, multi_gpu = True):
        self.net = net
        
        if self.coords:
            self.modelpath = 'weights/deeplabv3_coordconv.h5'
        elif self.net=='unet':
            self.modelpath = 'weights/unet.h5'
        else:
            self.modelpath = 'weights/deeplabv3.h5'

        if net == 'unet':
            model = create_unet(self.sz, len(get_VOC2012_classes()))
        else:   
            model = Deeplabv3(weights=None, input_tensor=None, 
                              input_shape = self.sz + (3,), classes=21, 
                              backbone='mobilenetv2', OS=16, alpha=1, use_coordconv = self.coords)
        if multi_gpu:
            from keras.utils import multi_gpu_model
            model = multi_gpu_model(deeplab_model, gpus=4)
        model.compile(optimizer = opt, 
                      loss = wisense_loss(w),
                      #loss = softmax_sparse_crossentropy_ignoring_last_label, 
                      metrics = [background_sparse_accuracy, foreground_sparse_accuracy, 'acc'])
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
        else:
            self.n_val = image_generator.n
    
        return zip(image_generator, mask_generator)

    def load_weights(self, model):
        model.load_weights(self.modelpath)
        
    def train_generator(self, model, train_generator, valid_generator, tf_board = False, mp = True):
        
        h = model.fit_generator(train_generator, 
                                steps_per_epoch=self.n_trn//self.batch_size, 
                                epochs = self.epochs, verbose=1, 
                                callbacks = self.build_callbacks(tf_board = tf_board, plot_process = False), 
                                validation_data=valid_generator, 
                                validation_steps=self.n_val//self.batch_size, 
                                class_weight=None, max_queue_size=10, 
                                workers=8, use_multiprocessing=mp)
        return h    
    
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
class PlotLearning(keras.callbacks.Callback, SegModel):
    def __init__(self):
        SegModel.__init__(self)
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.acc2 = []
        self.val_acc2 = []

#        self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('foreground_sparse_accuracy'))
        self.val_acc.append(logs.get('val_foreground_sparse_accuracy'))
        self.acc2.append(logs.get('background_sparse_accuracy'))
        self.val_acc2.append(logs.get('val_background_sparse_accuracy'))

        self.i += 1
        fig, ax1 = plt.subplots(figsize=(10,5));
        clear_output(wait=True)
        lns1 = ax1.plot(self.x, self.acc, label="Sparse accuracy foreground", color=[0,0,1], linewidth = 3.);
        lns2 = ax1.plot(self.x, self.val_acc, label="Sparse validation accuracy foreground", color=[0,0,0.5], linewidth = 3.);
        
        lns11 = ax1.plot(self.x, self.acc2, label="Sparse accuracy background", color=[0,1,0], linewidth = 3.);
        lns21 = ax1.plot(self.x, self.val_acc2, label="Sparse validation accuracy background", color=[0,0.5,0], linewidth = 3.);

        ax1.set_ylabel('Sparse accuracy',fontsize=15);
        ax1.set_xlabel('Epoch',fontsize=15);
        ax1.set_title('Deeplab V3+ Segmentation Model');
        ax2 = ax1.twinx()
        lns3 = ax2.plot(self.x, self.losses, '--', label="loss", color=[1,0,0]);
        lns4 = ax2.plot(self.x, self.val_losses, '--', label="validation loss", color=[0.5,0,0]);
        ax2.set_ylabel('Loss',fontsize=15)
        lns = lns1+lns2+lns3+lns4+lns11+lns21
        ax1.legend(lns, [l.get_label() for l in lns])
        ax1.grid(True)
        plt.xlim([-0.05, self.epochs+.05])
        plt.show();

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
