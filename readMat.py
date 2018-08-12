from scipy import io as sio
import re
import numpy as np


def squeeze(vars_):
    # Matlab save some params with shape (*, 1)
    # However, we don't need the trailing dimension in TensorFlow.
    if isinstance(vars_, (list, tuple)):
        return [np.squeeze(v, 1) for v in vars_]
    else:
        return np.squeeze(vars_, 1)


netparams = sio.loadmat('G:/py/SiamFC-TensorFlow/assets/2016-08-17.net.mat')['net']['params'][0][0]
params = dict()

for i in range(netparams.size):
    param = netparams[0][i]
    name = param["name"][0]
    value = param["value"]
    if value.ndim == 4:
        value = np.swapaxes(value, 0, 2)
        value = np.swapaxes(value, 1, 3)
        value = np.swapaxes(value, 0, 1)
    value_size = param["value"].shape[0]
    match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name, re.I)
    if match:
        items = match.groups()
    elif name == 'adjust_f':
        #params['detection/weights'] = squeeze(value)
        continue
    elif name == 'adjust_b':
        #params['detection/biases'] = squeeze(value)
        continue
    else:
        raise Exception('unrecognized layer params')

    op, layer, types = items
    layer = int(layer)
    if op == 'conv':
        if types == 'f':
            params['c%d' % layer] = value
        if types == 'b' and layer == 5:
            params['c5b'] = squeeze(value)
    elif op == 'bn':
        if types == 'm':
            params['s%d/g' % layer] = squeeze(value)
        elif types == 'b':
            params['s%d/b' % layer] = squeeze(value)
        elif types == 'x':
            m, v = squeeze(np.split(value, 2, 1))
            params['b%d/m' % layer] = m
            params['b%d/v' % layer] = np.square(v)

for key in params:
    name = key
    fname = name + '.prototxt'
    fname = fname.replace('/', '_')
    value = params[key]
    if value.ndim == 4:
        f = open(fname, 'w')
        vshape = value.shape[:]
        v_1d = value.reshape(value.shape[0]*value.shape[1]*value.shape[2]*value.shape[3])
        f.write('  blobs {\n')
        for vv in v_1d:
            f.write('    data: %8f' % vv)
            f.write('\n')
        f.write('    shape {\n')
        for s in vshape:
            f.write('      dim: ' + str(s))#print dims
            f.write('\n')
        f.write('    }\n')
        f.write('  }\n')
    elif value.ndim == 1 :#do not swap
        f = open(fname, 'w')
        f.write('  blobs {\n')
        for vv in value:
            f.write('    data: %.8f' % vv)
            f.write('\n')
        f.write('    shape {\n')
        f.write('      dim: ' + str(value.shape[0]))#print dims
        f.write('\n')
        f.write('    }\n')
        f.write('  }\n')
        f.close()













'''

  for i in range(netparams.size):
    param = netparams[0][i]
    name = param["name"][0]
    value = param["value"]
    value_size = param["value"].shape[0]

    match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name, re.I)
    if match:
      items = match.groups()
    elif name == 'adjust_f':
      params['detection/weights'] = squeeze(value)
      continue
    elif name == 'adjust_b':
      params['detection/biases'] = squeeze(value)
      continue
    else:
      raise Exception('unrecognized layer params')

    op, layer, types = items
    layer = int(layer)
    if layer in [1, 3]:
      if op == 'conv':  # convolution
        if types == 'f':
          params['conv%d/weights' % layer] = value
        elif types == 'b':
          value = squeeze(value)
          params['conv%d/biases' % layer] = value
      elif op == 'bn':  # batch normalization
        if types == 'x':
          m, v = squeeze(np.split(value, 2, 1))
          params['conv%d/BatchNorm/moving_mean' % layer] = m
          params['conv%d/BatchNorm/moving_variance' % layer] = np.square(v)
        elif types == 'm':
          value = squeeze(value)
          params['conv%d/BatchNorm/gamma' % layer] = value
        elif types == 'b':
          value = squeeze(value)
          params['conv%d/BatchNorm/beta' % layer] = value
      else:
        raise Exception
    elif layer in [2, 4]:
      if op == 'conv' and types == 'f':
        b1, b2 = np.split(value, 2, 3)
      else:
        b1, b2 = np.split(value, 2, 0)
      if op == 'conv':
        if types == 'f':
          params['conv%d/b1/weights' % layer] = b1
          params['conv%d/b2/weights' % layer] = b2
        elif types == 'b':
          b1, b2 = squeeze(np.split(value, 2, 0))
          params['conv%d/b1/biases' % layer] = b1
          params['conv%d/b2/biases' % layer] = b2
      elif op == 'bn':
        if types == 'x':
          m1, v1 = squeeze(np.split(b1, 2, 1))
          m2, v2 = squeeze(np.split(b2, 2, 1))
          params['conv%d/b1/BatchNorm/moving_mean' % layer] = m1
          params['conv%d/b2/BatchNorm/moving_mean' % layer] = m2
          params['conv%d/b1/BatchNorm/moving_variance' % layer] = np.square(v1)
          params['conv%d/b2/BatchNorm/moving_variance' % layer] = np.square(v2)
        elif types == 'm':
          params['conv%d/b1/BatchNorm/gamma' % layer] = squeeze(b1)
          params['conv%d/b2/BatchNorm/gamma' % layer] = squeeze(b2)
        elif types == 'b':
          params['conv%d/b1/BatchNorm/beta' % layer] = squeeze(b1)
          params['conv%d/b2/BatchNorm/beta' % layer] = squeeze(b2)
      else:
        raise Exception

    elif layer in [5]:
      if op == 'conv' and types == 'f':
        b1, b2 = np.split(value, 2, 3)
      else:
        b1, b2 = squeeze(np.split(value, 2, 0))
      assert op == 'conv', 'layer5 contains only convolution'
      if types == 'f':
        params['conv%d/b1/weights' % layer] = b1
        params['conv%d/b2/weights' % layer] = b2
      elif types == 'b':
        params['conv%d/b1/biases' % layer] = b1
        params['conv%d/b2/biases' % layer] = b2
'''