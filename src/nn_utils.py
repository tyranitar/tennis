from torch import nn
import math

def init_orth(layer, w_scale=1):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)

    return layer

def init_uniform(layer, lim):
    layer.weight.data.uniform_(-lim, lim)

    return layer

def init_sqrt_fan_in(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1 / math.sqrt(fan_in)

    return init_uniform(layer, lim)

def init_kaiming(layer, mode="fan_in", nonlin="relu", w_scale=1):
    nn.init.kaiming_normal_(layer.weight.data, mode=mode, nonlinearity=nonlin)
    layer.weight.data.mul_(w_scale)

    return layer

init_default = init_orth

def get_fc_layer(in_dim, out_dim):
    return nn.Sequential(
        init_default(nn.Linear(in_dim, out_dim)),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    )

def get_fc_layers(layer_dims):
    return [
        get_fc_layer(layer_dims[i], layer_dims[i + 1])
        for i in range(len(layer_dims) - 1)
    ]

def ensure_tuple(x, n):
    if type(x) == tuple:
        if len(x) != n:
            raise Exception(f"expected tuple of size {n}.")

        return x
    else:
        return tuple(x for _ in range(n))

def assert_valid_conv_layer_dims(input_chan_size, layer_dims):
    prev_chan_size = ensure_tuple(input_chan_size, 2)

    for _, kernel_size, stride in layer_dims:
        kernel_size = ensure_tuple(kernel_size, 2)
        stride = ensure_tuple(stride, 2)

        for i in range(2):
            assert prev_chan_size[i] >= kernel_size[i]
            assert stride[i] > 0
            assert (prev_chan_size[i] - kernel_size[i]) % stride[i] == 0

        prev_chan_size = tuple(
            int((prev_chan_size[i] - kernel_size[i]) / stride[i] + 1)
            for i in range(2)
        )

    # Final output size.
    return layer_dims[-1][0] * prev_chan_size[0] * prev_chan_size[1]

def get_conv_layers(num_input_chans, layer_dims):
    get_conv_args = lambda i: \
        (layer_dims[i - 1][0], *layer_dims[i]) \
        if i > 0 else \
        (num_input_chans, *layer_dims[i])

    # NOTE: No pooling.
    return [
        nn.Sequential(
            init_default(nn.Conv2d(*get_conv_args(i))),
            nn.BatchNorm2d(layer_dims[i][0]),
            nn.ReLU(),
        )
        for i in range(len(layer_dims))
    ]
