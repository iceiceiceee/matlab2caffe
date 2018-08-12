import caffe
from caffe import layers as L, params as P, to_proto


root = 'G:/py/tf2caffe/'
deploy = root+'inference.prototxt'


def conv_relu(bottom, kernel_size, num_output, stride=1, pad=0, weight_filler=dict(type='xavier'), group=1):
    conv = L.Convolution(bottom=bottom, kernel_size=kernel_size, stride=stride, num_output=num_output, pad=pad,
                         weight_filler=weight_filler, group=group)
    relu = L.Relu(conv, in_place=True)
    return conv, relu


def max_pool(bottom, kernel_size, stride):


def create_deploy(img_list, batch_size, include_acc=False):
    n = caffe.NetSpec()
    n.data = L.ImageData(source=img_list, batch_size=batch_size, ntop=2, root_folder=root, transform_param=dict(scale=0.00390625))
    n.conv1, n.relu1 = conv_relu(n.data, kernel_size=11, stride=2, num_output=96, weight_filler=dict(type='uniform'))

    pool1=L.Pooling(conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    conv2=L.Convolution(pool1, kernel_size=5, stride=1,num_output=50, pad=0,weight_filler=dict(type='xavier'))
    pool2=L.Pooling(conv2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    fc3=L.InnerProduct(pool2, num_output=500,weight_filler=dict(type='xavier'))
    relu3=L.ReLU(fc3, in_place=True)
    fc4 = L.InnerProduct(relu3, num_output=10,weight_filler=dict(type='xavier'))
    #最后没有accuracy层，但有一个Softmax层
    prob=L.Softmax(fc4)
    return to_proto(prob)