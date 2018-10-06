import tensorflow as tf
import cv2
import os
import numpy as np

os.system("sudo nvpmodel -m 0")
os.system("sudo ~/jetson_clocks.sh")

def open_onboard_camera():
    return cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

usbcam = 0
on_image = False
if on_image:
    img_name = 'dog.jpg'

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


colorsFile = "colors.txt";
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')

colors = [] #[0,0,0]
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = .5
fontColor              = (255,255,255)
lineType               = 1

h = 256
colorbar = np.ones((h, 150, 3), dtype='uint8')
s = h//len(colors)
ok_length = s*len(colors)
for k, i in enumerate(range(0, ok_length, s)):
    colorbar[i:i+s, :, :] = colors[k]
    if colors[k].mean() < 127:
        c = (255,255,255)
    else:
        c = (0,0,0)
    cv2.putText(colorbar, PASCAL_VOC_classes[k], 
                (2, i+round(s*0.7)), 
                cv2.FONT_ITALIC, 
                0.5,
                c,
                2);


colorbar = colorbar[:ok_length, :, :]
colorbar = cv2.resize(colorbar, (150, h))

if not on_image:
    if usbcam:
        cap = cv2.VideoCapture(usbcam)
    else:
        cap = open_onboard_camera()
        
graph = tf.Graph()

output_node = 'bilinear_upsampling_2/ResizeBilinear' # mrcnn_mask_2/Reshape_1, bilinear_upsampling_2/ResizeBilinear, detection_masks

if on_image:
    original_frame = cv2.imread(img_name)
    frame = cv2.resize(original_frame, (h,h))
    image = frame.astype('float32')
    image /= 255.0
    image = 2*image - 1
    image = np.expand_dims(image, axis=0)

with graph.as_default():
    with tf.Session(graph=graph) as sess:
        with tf.gfile.FastGFile('model_tf.pb', 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())

        input_var = tf.placeholder("float32",[1,h,h,3])
        [output_image] = tf.import_graph_def(graph_def,input_map={'input_1:0':input_var},return_elements=[output_node + ':0'], name='')

        if on_image:
            pred = sess.run(output_image, feed_dict={input_var:image})

        while(True):
            if on_image:
                break
            ret, original_frame = cap.read()
            #if not usbcam:
            #    original_frame = original_frame[60:-60,:,:]
            #    original_frame = original_frame[:,:360,:]


            frame = cv2.resize(original_frame, (h,h))
            
            image = frame.astype('float32')
            image /= 255.0
            
            image = 2*image - 1
            image = np.expand_dims(image, axis=0)
            pred = sess.run(output_image, feed_dict={input_var:image})
            p = np.argmax(pred, axis=-1).astype('int')
            x = [colors[i] for i in p[0].flatten()]
            mask = np.reshape(np.stack(x),(h,h,3))
            added_image = cv2.addWeighted(frame,0.5,mask.astype('uint8'),0.5,0)
            classes = [PASCAL_VOC_classes[c] for c in np.unique(p[0])]

            cv2.putText(added_image, str(np.stack(classes))[2:-2], 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)
            
            seg_img = np.concatenate((added_image, colorbar), axis=1)
            seg_img = cv2.resize(seg_img, (original_frame.shape[1], original_frame.shape[0]))
            cv2.imshow('frame',seg_img)
            #cv2.imshow('frame',frame*2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# When everything done, release the capture
if not on_image:
    cap.release()
    cv2.destroyAllWindows()

else:

    p = np.argmax(pred, axis=-1).astype('int')
    x = [colors[i] for i in p[0].flatten()]
    mask = np.reshape(np.stack(x),(h,h,3))
    added_image = cv2.addWeighted(frame,0.5,mask.astype('uint8'),0.5,0)

    classes = [PASCAL_VOC_classes[c] for c in np.unique(p[0])]
    cv2.putText(added_image, str(np.stack(classes))[2:-2], bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    seg_img = np.concatenate((added_image, colorbar), axis=1)
    cv2.imshow('frame',seg_img)
    cv2.waitKey(10)

