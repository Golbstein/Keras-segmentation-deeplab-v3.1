from sklearn.model_selection import train_test_split
from deeplabv3p import preprocess_input
import glob
from utils import *

PATH = '/workspace/datasets/VOCdevkit/VOC2012/'

image_size = (512, 512)

if len(glob.glob(PATH+'*.npy')):
    X_train = np.load(PATH + 'X_train.npy')
    X_valid = np.load(PATH + 'X_valid.npy')
    X_test = np.load(PATH + 'X_test.npy')
    y_train = np.load(PATH + 'y_train.npy')
    y_valid = np.load(PATH + 'y_valid.npy')
    y_test = np.load(PATH + 'y_test.npy')
else:
    X, y = load_train_data(PATH, image_size)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=29)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=29)
    #np.save(PATH+'X_train', X_train)
    #np.save(PATH+'X_valid', X_valid)
    #np.save(PATH+'X_test', X_test)
    #np.save(PATH+'y_train', y_train)
    #np.save(PATH+'y_valid', y_valid)
    #np.save(PATH+'y_test', y_test)

print('train size: ', X_train.shape)
print('validation size: ', X_valid.shape)
print('test size: ', X_test.shape)


deeplab_seg = SegModel(PATH, image_size, use_coords = True)
deeplab_seg.set_batch_size(16)
deeplab_seg.set_num_epochs(500)
opt = SGD(lr=0.01, momentum = 0.9)
deeplab_model = deeplab_seg.create_seg_model(opt, net='deeplabv3', load_weights = None, multi_gpu = False)
h = deeplab_seg.train(deeplab_model, X_train, y_train, (X_valid, y_valid), plot_train_process = False)
