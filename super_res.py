import tensorflow as tf
import cv2
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def preprocess_input(x):
    b_mean = 104.00698793
    g_mean = 116.66876762
    r_mean = 122.67891434
    x = x.astype(np.float32)
    x[:, :, 0] -= b_mean
    x[:, :, 1] -= g_mean
    x[:, :, 2] -= r_mean
    return x


def mousePosition(event,x,y,flags,param):
    global c1, c2
    if event == cv2.EVENT_LBUTTONDOWN:
        c1 = x
        c2 = y

c1 = 150
c2 = 260

h = 48

w = 672

os.system("sudo nvpmodel -m 0")
os.system("sudo ~/jetson_clocks.sh")

scale = 4

cap = cv2.VideoCapture(0)

graph = tf.Graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
output_node = 'lambda_78/concat' 
with graph.as_default():
    with tf.Session(graph=graph, config=config) as sess:
        with tf.gfile.FastGFile('x4.pb', 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
        input_var = tf.placeholder("float32",[1,h,h,3])
        [output_image] = tf.import_graph_def(graph_def,input_map={'input_1:0':input_var},
                                             return_elements=[output_node + ':0'], name='')

        cv2.namedWindow('frame', cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('frame', mousePosition)
        
        while(True):
            
            X_low = c1-h//2
            X_high = c1+h//2
            Y_low = c2-h//2
            Y_high = c2+h//2

            ret, original_frame = cap.read()
            original_frame = original_frame[:,:w,:]
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YUV)
            original_frame[:,:,0] = clahe.apply(original_frame[:,:,0])
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_YUV2BGR)
            original_frame = cv2.resize(original_frame, (original_frame.shape[1], h*scale*2))

            crop = original_frame[Y_low:Y_high, X_low:X_high, :]
            cropp = cv2.resize(crop, (int(h//1.2), int(h//1.2)), interpolation=cv2.INTER_NEAREST)
            h_crop = cv2.resize(cropp, (scale*h, scale*h), interpolation=cv2.INTER_NEAREST)
            x = preprocess_input(crop)
            sr = sess.run(output_image, feed_dict={input_var:x[None]})
            out = np.clip(sr[0], 0.0, 255.0)
            out = out.astype(np.uint8)
            
            up_sample = np.concatenate((h_crop, out), axis=0)
            full_img = np.concatenate((original_frame, up_sample), axis=1)
            
            cv2.rectangle(full_img,(X_low,Y_low),(X_high,Y_high), (0,255,0), 2)
            
            cv2.putText(full_img, 'Original', 
                        (full_img.shape[1]-scale*h+7,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,
                        (0,0,255),
                        3)
            cv2.putText(full_img, 'X{} - SR'.format(scale), 
                        (full_img.shape[1]-scale*h+7, 30+full_img.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,
                        (0,0,255),
                        3)
            cv2.imshow('frame',full_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

            
