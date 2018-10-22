
from utils import *
import pickle
PATH = '/workspace/datasets/OpenSourceDatasets/VOCdevkit/VOC2012/'
image_size = (376, 672)
batch_size = 4
seed = 7
validation_split = .2
n_classes = len(get_VOC2012_classes())-1
epochs = 1000
NET = 'deeplab'
to_compile = False

SegClass = SegModel(PATH, image_size, use_coords = True)
SegClass.set_batch_size(batch_size)
SegClass.set_num_epochs(epochs)

opt = SGD(lr = 7e-3, momentum=0.3)

if NET == 'icnet':
    to_compile = False
    
model = SegClass.create_seg_model(opt, net=NET, load_weights = False, 
                                  multi_gpu = False, to_compile = to_compile, alpha = 1)

#model.load_weights('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5')

if not to_compile:
    model.compile(optimizer = opt, sample_weight_mode = "temporal",
              loss = 'categorical_crossentropy',
              metrics = [Mean_IOU, background_accuracy, 
                         accuracy_ignoring_last_label,
                         foreground_accuracy])
#     model.compile(optimizer = opt, sample_weight_mode = "temporal",
#                   loss_weights=[1.0, 0.4, 0.16],
#                   loss = 'categorical_crossentropy',
#                   metrics = [Mean_IOU, background_accuracy, 
#                   accuracy_ignoring_last_label,
#                   foreground_accuracy])

train_generator = SegClass.create_generators(resize_shape = tuple(reversed(image_size)), blur = 5,
                                             crop_shape = False, mode = 'train',n_classes = 21, 
                                             horizontal_flip = True, vertical_flip = False, 
                                             icnet = NET == 'icnet',brightness=0.3, rotation=2.0, 
                                             zoom=0.1, validation_split = .2, seed = 7, do_ahisteq = True)

valid_generator = SegClass.create_generators(resize_shape = tuple(reversed(image_size)),
                                             crop_shape = False, mode = 'validation', n_classes = 21, 
                                             horizontal_flip = True, vertical_flip = False, 
                                             icnet = NET == 'icnet',brightness=False, rotation=False, 
                                             zoom=0, validation_split = .2, seed = 7, do_ahisteq = True)


h = SegClass.train_generator(model, train_generator = train_generator,
                             valid_generator = valid_generator, plot_test_images = False,
                             tf_board = True, mp = True)


#lr_decay = LearningRateScheduler(PolyDecay(0.007, 0.9, epochs).scheduler, verbose = 1)

#h = model.fit_generator(train_generator, len(train_generator), epochs, callbacks=[checkpoint, tensorboard, lr_decay], 
#                       validation_data=val_generator, validation_steps=len(val_generator), workers=workers, 
#                       use_multiprocessing=True, shuffle=True, max_queue_size=10, verbose=1)

with open(NET+'_'+str(image_size[0])+'.pkl', 'wb') as f:
    pickle.dump(h.history, f, pickle.HIGHEST_PROTOCOL)
