from utils import *
from icnet import build_bn
PATH = '/workspace/datasets/OpenSourceDatasets/VOCdevkit/VOC2012/'
resize_shape=(512, 512)

batch_size = 16
seed = 7
validation_split = .25
n_classes = len(get_VOC2012_classes())-1
epochs = 300
NET = 'icnet'
to_compile = True

SegClass = SegModel(PATH, image_size, use_coords = False)
SegClass.set_batch_size(2)
SegClass.set_num_epochs(100)

opt = SGD(lr=0.007, momentum=0.9)

if NET == 'icnet':
    to_compile = False
    
model = SegClass.create_seg_model(opt, net=NET, load_weights = False, 
                                  multi_gpu = False, to_compile = to_compile, alpha = 1)
if not to_compile:
    model.compile(optimizer = optim, sample_weight_mode = "temporal",
                  loss_weights=[1.0, 0.4, 0.16],
                  loss = 'categorical_crossentropy',
                  metrics = [Mean_IOU, background_sparse_accuracy, 
                  sparse_accuracy_ignoring_last_label,
                  foreground_sparse_accuracy])


train_generator = SegClass.create_generators(resize_shape = image_size,
                                             crop_shape = False, mode = 'train',n_classes = 21, 
                                             horizontal_flip = True, vertical_flip = False, 
                                             icnet = NET == 'icnet',brightness=0.1, rotation=2.0, 
                                             zoom=0.1, validation_split = validation_split, seed = 7)

valid_generator = SegClass.create_generators(resize_shape = image_size,
                                             crop_shape = False, mode = 'validation', n_classes = 21, 
                                             horizontal_flip = True, vertical_flip = False, 
                                             icnet = NET == 'icnet',brightness=0, rotation=0, 
                                             zoom=0, validation_split = validation_split, seed = 7)


h, cyclr = SegClass.train_generator(model, train_generator=train_generator,
                                    valid_generator = valid_generator,
                                    tf_board = True, mp = False)

#lr_decay = LearningRateScheduler(PolyDecay(0.007, 0.9, epochs).scheduler, verbose = 1)

#h = model.fit_generator(train_generator, len(train_generator), epochs, callbacks=[checkpoint, tensorboard, lr_decay], 
#                       validation_data=val_generator, validation_steps=len(val_generator), workers=workers, 
#                       use_multiprocessing=True, shuffle=True, max_queue_size=10, verbose=1)

with open(NET+'_'+str(resize_shape[0])+'.pkl', 'wb') as f:
    pickle.dump(h.history, f, pickle.HIGHEST_PROTOCOL)
