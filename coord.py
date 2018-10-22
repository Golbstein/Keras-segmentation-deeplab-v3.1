from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


class _CoordinateChannel(Layer):
    """ Adds Coordinate Channels to the input tensor.
    # Arguments
        rank: An integer, the rank of the input data-uniform,
            e.g. "2" for 2D convolution.
        use_radius: Boolean flag to determine whether the
            radius coordinate should be added for 2D rank
            inputs or not.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        ND tensor with shape:
        `(samples, channels, *)`
        if `data_format` is `"channels_first"`
        or ND tensor with shape:
        `(samples, *, channels)`
        if `data_format` is `"channels_last"`.
    # Output shape
        ND tensor with shape:
        `(samples, channels + 2, *)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, *, channels + 2)`
        if `data_format` is `"channels_last"`.
    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, rank,
                 use_radius=False,
                 data_format=None,
                 **kwargs):
        super(_CoordinateChannel, self).__init__(**kwargs)

        if data_format not in [None, 'channels_first', 'channels_last']:
            raise ValueError('`data_format` must be either "channels_last", "channels_first" '
                             'or None.')

        self.rank = rank
        self.use_radius = use_radius
        self.data_format = K.image_data_format() if data_format is None else data_format
        self.axis = 1 if K.image_data_format() == 'channels_first' else -1

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[self.axis]

        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={self.axis: input_dim})
        self.built = True

    def call(self, inputs, training=None, mask=None):
        input_shape = K.shape(inputs)

        if self.rank == 1:
            input_shape = [input_shape[i] for i in range(3)]
            batch_shape, dim, channels = input_shape

            xx_range = K.tile(K.expand_dims(K.arange(0, dim), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=-1)

            xx_channels = K.cast(xx_range, K.floatx())
            xx_channels = xx_channels / K.cast(dim - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            outputs = K.concatenate([inputs, xx_channels], axis=-1)

        if self.rank == 2:
            if self.data_format == 'channels_first':
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 1])
                input_shape = K.shape(inputs)

            input_shape = [input_shape[i] for i in range(4)]
            batch_shape, dim1, dim2, channels = input_shape

            xx_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = K.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)
            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            yy_ones = K.ones(K.stack([batch_shape, dim1]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = K.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim1 - 1, K.floatx())
            xx_channels = (xx_channels * 2) - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim2 - 1, K.floatx())
            yy_channels = (yy_channels * 2) - 1.

            outputs = K.concatenate([inputs, xx_channels, yy_channels], axis=-1)

            if self.use_radius:
                rr = K.sqrt(K.square(xx_channels - 0.5) +
                            K.square(yy_channels - 0.5))
                outputs = K.concatenate([outputs, rr], axis=-1)

            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])

        if self.rank == 3:
            if self.data_format == 'channels_first':
                inputs = K.permute_dimensions(inputs, [0, 2, 3, 4, 1])
                input_shape = K.shape(inputs)

            input_shape = [input_shape[i] for i in range(5)]
            batch_shape, dim1, dim2, dim3, channels = input_shape

            xx_ones = K.ones(K.stack([batch_shape, dim3]), dtype='int32')
            xx_ones = K.expand_dims(xx_ones, axis=-1)

            xx_range = K.tile(K.expand_dims(K.arange(0, dim2), axis=0),
                              K.stack([batch_shape, 1]))
            xx_range = K.expand_dims(xx_range, axis=1)

            xx_channels = K.batch_dot(xx_ones, xx_range, axes=[2, 1])
            xx_channels = K.expand_dims(xx_channels, axis=-1)
            xx_channels = K.permute_dimensions(xx_channels, [0, 2, 1, 3])

            xx_channels = K.expand_dims(xx_channels, axis=1)
            xx_channels = K.tile(xx_channels,
                                 [1, dim1, 1, 1, 1])

            yy_ones = K.ones(K.stack([batch_shape, dim2]), dtype='int32')
            yy_ones = K.expand_dims(yy_ones, axis=1)

            yy_range = K.tile(K.expand_dims(K.arange(0, dim3), axis=0),
                              K.stack([batch_shape, 1]))
            yy_range = K.expand_dims(yy_range, axis=-1)

            yy_channels = K.batch_dot(yy_range, yy_ones, axes=[2, 1])
            yy_channels = K.expand_dims(yy_channels, axis=-1)
            yy_channels = K.permute_dimensions(yy_channels, [0, 2, 1, 3])

            yy_channels = K.expand_dims(yy_channels, axis=1)
            yy_channels = K.tile(yy_channels,
                                 [1, dim1, 1, 1, 1])

            zz_range = K.tile(K.expand_dims(K.arange(0, dim1), axis=0),
                              K.stack([batch_shape, 1]))
            zz_range = K.expand_dims(zz_range, axis=-1)
            zz_range = K.expand_dims(zz_range, axis=-1)

            zz_channels = K.tile(zz_range,
                                 [1, 1, dim2, dim3])
            zz_channels = K.expand_dims(zz_channels, axis=-1)

            xx_channels = K.cast(xx_channels, K.floatx())
            xx_channels = xx_channels / K.cast(dim2 - 1, K.floatx())
            xx_channels = xx_channels * 2 - 1.

            yy_channels = K.cast(yy_channels, K.floatx())
            yy_channels = yy_channels / K.cast(dim3 - 1, K.floatx())
            yy_channels = yy_channels * 2 - 1.

            zz_channels = K.cast(zz_channels, K.floatx())
            zz_channels = zz_channels / K.cast(dim1 - 1, K.floatx())
            zz_channels = zz_channels * 2 - 1.

            outputs = K.concatenate([inputs, zz_channels, xx_channels, yy_channels],
                                    axis=-1)

            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 4, 1, 2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[self.axis]

        if self.use_radius and self.rank == 2:
            channel_count = 3
        else:
            channel_count = self.rank

        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] + channel_count
        return tuple(output_shape)

    def get_config(self):
        config = {
            'rank': self.rank,
            'use_radius': self.use_radius,
            'data_format': self.data_format
        }
        base_config = super(_CoordinateChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoordinateChannel1D(_CoordinateChannel):
    """ Adds Coordinate Channels to the input tensor of rank 1.
    # Arguments
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`
    # Output shape
        3D tensor with shape: `(batch_size, steps, input_dim + 2)`
    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, data_format=None, **kwargs):
        super(CoordinateChannel1D, self).__init__(
            rank=1,
            use_radius=False,
            data_format=data_format,
            **kwargs
        )

    def get_config(self):
        config = super(CoordinateChannel1D, self).get_config()
        config.pop('rank')
        config.pop('use_radius')
        return config


class CoordinateChannel2D(_CoordinateChannel):
    """ Adds Coordinate Channels to the input tensor.
    # Arguments
        use_radius: Boolean flag to determine whether the
            radius coordinate should be added for 2D rank
            inputs or not.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)`
        if `data_format` is `"channels_last"`.
    # Output shape
        4D tensor with shape:
        `(samples, channels + 2/3, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels + 2/3)`
        if `data_format` is `"channels_last"`.
        If `use_radius` is set, then will have 3 additional filers,
        else only 2 additional filters will be added.
    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, use_radius=False,
                 data_format=None,
                 **kwargs):
        super(CoordinateChannel2D, self).__init__(
            rank=2,
            use_radius=use_radius,
            data_format=data_format,
            **kwargs
        )

    def get_config(self):
        config = super(CoordinateChannel2D, self).get_config()
        config.pop('rank')
        return config


class CoordinateChannel3D(_CoordinateChannel):
    """ Adds Coordinate Channels to the input tensor.
    # Arguments
        rank: An integer, the rank of the input data-uniform,
            e.g. "2" for 2D convolution.
        use_radius: Boolean flag to determine whether the
            radius coordinate should be added for 2D rank
            inputs or not.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)`
        if `data_format` is `"channels_last"`.
    # Output shape
        5D tensor with shape:
        `(samples, channels + 2, conv_dim1, conv_dim2, conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels + 2)`
        if `data_format` is `"channels_last"`.
    # References:
        - [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    """

    def __init__(self, data_format=None,
                 **kwargs):
        super(CoordinateChannel3D, self).__init__(
            rank=3,
            use_radius=False,
            data_format=data_format,
            **kwargs
        )

    def get_config(self):
        config = super(CoordinateChannel3D, self).get_config()
        config.pop('rank')
        config.pop('use_radius')
        return config


get_custom_objects().update({'CoordinateChannel1D': CoordinateChannel1D,
                             'CoordinateChannel2D': CoordinateChannel2D,
                             'CoordinateChannel3D': CoordinateChannel3D})