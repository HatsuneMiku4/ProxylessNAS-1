# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import yaml
import os
import sys
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

from itertools import product

import numpy as np


def download_url(url, model_dir='~/.torch/proxyless_nas', overwrite=False):
    target_dir = url.split('//')[-1]
    target_dir = os.path.dirname(target_dir)
    model_dir = os.path.expanduser(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file


class LatencyEstimator(object):
    def __init__(self, url='https://hanlab.mit.edu/files/proxylessNAS/LatencyTools/mobile_trim.yaml'):
        fname = download_url(url, overwrite=False)  # True

        with open(fname, 'r') as fp:
            # Use of PyYAML’s yaml.load function without specifying the Loader=… parameter, has been deprecated.
            self.lut = yaml.load(fp, Loader=yaml.FullLoader)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, ltype: str, _input, output, expand=None, kernel=None, stride=None, idskip=None, ):
        """
        :param ltype:
            Layer type must be one of the followings
                1. `Conv`: The initial 3x3 conv with stride 2.
                2. `Conv_1`: The upsample 1x1 conv that increases num_filters by 4 times.
                3. `Logits`: All operations after `Conv_1`.
                4. `expanded_conv`: MobileInvertedResidual
        :param _input: input shape (h, w, #channels)
        :param output: output shape (h, w, #channels)
        :param expand: expansion ratio
        :param kernel: kernel size
        :param stride:
        :param idskip: indicate whether has the residual connection
        """
        infos = [ltype, 'input:%s' % self.repr_shape(_input), 'output:%s' % self.repr_shape(output), ]

        if ltype in ('expanded_conv',):
            assert None not in (expand, kernel, stride, idskip)
            infos += ['expand:%d' % expand, 'kernel:%d' % kernel, 'stride:%d' % stride, 'idskip:%d' % idskip]
        key = '-'.join(infos)
        return self.lut[key]['mean']


class FPGALatencyEstimator(LatencyEstimator):

    BATCHSIZE = 16
    BANDWIDTH = 2.656e9
    FREQUENCE = 1.66e8
    DEPTH = 5
    II = 1
    T_M = T_N = 64
    P_W_RANGE = (3, 7+1)
    P_M_RANGE = P_N_RANGE = (8, 64+1)

    def __init__(self):
        super(FPGALatencyEstimator, self).__init__()
        del self.lut

    def predict(self, ltype: str, _input, output, expand=None, kernel=None, stride=None, idskip=None):
        if ltype in ['Conv', 'Conv_1']:
            kernel, stride = (1, 1) if ltype == 'Conv_1' else (3, 2)
            return self.conv_layer_sweep_hw_params(_input, _input, kernel, stride)[0]
        elif ltype == 'Logits':
            return self.conv_layer_sweep_hw_params([1, 1, _input[-1]], [1, 1, output[-1]], 1)[0]
        else:
            return self.mbconv_layer_sweep_hw_params(_input, _input, kernel, stride, expand)[0]

    @staticmethod
    def conv_layer_latency(H_in, W_in, M, H_out, W_out, N, r, P_w, P_m, P_n,
                           T_m=T_M, T_n=T_N, bandwidth=BANDWIDTH, freq=FREQUENCE, II=II,
                           depth=DEPTH, group=False, stride=1):
        if not group and r > 1 and stride > 1:
            T_compute = (W_in / stride * np.ceil(T_m / P_m) * np.ceil(T_n / P_n) * II * r + depth) / freq
        elif not group and r > 1:
            T_compute = (np.ceil(W_in / P_w) * np.ceil(T_m / P_m) * np.ceil(T_n / P_n) * II * r + depth) / freq
        else:
            T_compute = (W_in * np.ceil(T_n / P_n) * II * r * r + depth) / freq
        T_transfer = (max(W_out * min(T_n, N), W_in * min(T_m, M) * stride) * 2) / bandwidth
        T_init = ((T_m * T_n * r * r * 2) + (W_in * T_m * r * 2)) / bandwidth
        T_init /= FPGALatencyEstimator.BATCHSIZE
        T_iteration = max(T_compute, T_transfer) + T_init
        T_total = H_out * np.ceil(M / T_m) * np.ceil(N / T_n) * T_iteration
        return T_total

    @staticmethod
    def conv_layer_sweep_hw_params(in_shape, out_shape, kernel_size=3, stride=1, is_dwconv=False):
        global P_W_RANGE, P_N_RANGE, P_M_RANGE
        best_lat, best_param = 10000, dict.fromkeys(['P_w', 'P_n', 'P_m'])
        for P_w, P_n, P_m in product(range(*P_W_RANGE), range(*P_N_RANGE), range(*P_M_RANGE)):
            lat = FPGALatencyEstimator.conv_layer_latency(
                *in_shape, *out_shape, r=kernel_size, stride=stride, group=is_dwconv,
                P_w=P_w, P_n=P_n, P_m=P_m)
            if lat < best_lat:
                best_lat = lat
                best_param['P_w'] = P_w
                best_param['P_m'] = P_m
                best_param['P_n'] = P_n

        return best_lat, best_param

    @staticmethod
    def mbconv_layer_sweep_hw_params(in_shape, out_shape, kernel_size=3, stride=1, expand_ratio=6):
        (H_in, W_in, C_in), (H_out, W_out, C_out) = in_shape, out_shape
        feature_dim = round(C_in * expand_ratio)
        H_pad, W_pad = H_in + kernel_size - 1, W_in + kernel_size - 1
        H_out, W_out = H_in // stride, W_in // stride
        latency = 0
        best_params = dict.fromkeys(['inverted_bottleneck', 'depth_conv', 'point_linear'])

        if expand_ratio != 1:
            lat, param = FPGALatencyEstimator.conv_layer_sweep_hw_params(
                [H_in, W_in, C_in], [H_in, W_in, feature_dim], kernel_size=1)
            latency, best_params['inverted_bottleneck'] = latency + lat, param
        lat, param = FPGALatencyEstimator.conv_layer_sweep_hw_params(
            [H_pad, W_pad, feature_dim], [H_out, W_out, 1], kernel_size=kernel_size, stride=stride, is_dwconv=True)
        latency, best_params['depth_conv'] = latency + lat, param
        lat, param = FPGALatencyEstimator.conv_layer_sweep_hw_params(
            [H_out, W_out, feature_dim], [H_out, W_out, C_out], kernel_size=1)
        latency, best_params['point_linear'] = latency + lat, param

        return latency, best_params


if __name__ == '__main__':
    est = LatencyEstimator()
    s = est.predict('expanded_conv', _input=(112, 112, 16), output=(56, 56, 24), expand=3, kernel=5, stride=2, idskip=0)
    print(s)
