import caffe
import cv2
import scipy.io as sio
import numpy as np

root = 'G:/py/tf2caffe/genmodel/'
deploy = root + 'alex.prototxt'
caffe_model = root + 'alexnet.caffemodel'


img = cv2.imread('G:/py/SiamFC-TensorFlow/tests/01.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)
img = np.reshape(img, [1, 3, 360, 480])

net = caffe.Net(deploy, caffe_model, caffe.TEST)
net.blobs['data'].data[...] = img



feature_map = net.forward()
feature_map = feature_map['conv5']
feature = np.squeeze(feature_map, 0)
feature = np.swapaxes(feature, 0, 2)
feature = np.swapaxes(feature, 0, 1)

ideal_feature = sio.loadmat('G:/py/SiamFC-TensorFlow/tests/result.mat')['r']['z_features'][0][0]

diff = feature - ideal_feature
diff = np.sqrt(np.mean(np.square(diff)))
print('Feature computation difference: {}'.format(diff))
print('You should get something like: 0.00892720464617')