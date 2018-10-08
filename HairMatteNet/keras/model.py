from keras.models import Input, Model
from keras.layers import Conv2D, DepthwiseConv2D
from keras.layers import UpSampling2D, BatchNormalization
from keras.activations import relu
from config import get_config

config = get_config()

class HairMetteNet(Model):
    def __init__(self, org_shape):
        self.nf = config.nf
        def _Layer_Depwise_Encode(input, out_channels, stride=2, reserve=False):
            if reserve == True:
                stride = 1
            net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(input)
            net = BatchNormalization(momentum=0.99, epsilon=1e-5)(net)
            net = relu(net)
            net = Conv2D(filters=out_channels, kernel_size=1, strides=stride)(net)
            net = relu(net)
            return net

        def _Layer_Depwise_Dncode(input, out_channels):
            net = DepthwiseConv2D(kernel_size=3, stride=1, padding='same')(input)
            net = Conv2D(filters=out_channels, kernel_size=1, stride=1, activation='relu', )(net)
            return net

        def encode_layers1(input):
            net = _Layer_Depwise_Encode(input=input, out_channels=self.nf)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*2)
            return net

        def encode_layers2(input):
            net = _Layer_Depwise_Encode(input=input, out_channels=self.nf*4)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*4)
            return net


        def encode_layers3(input):
            net = _Layer_Depwise_Encode(input=input, out_channels=self.nf*8)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*8)
            return net


        def encode_layers4(input):
            net = _Layer_Depwise_Encode(input=input, out_channels=self.nf*16)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*16)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*16)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*16)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*16)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*16)
            return net

        def encode_layers5(input):
            net = _Layer_Depwise_Encode(input=input, out_channels=self.nf*32)
            net = _Layer_Depwise_Encode(input=net, out_channels=self.nf*32)
            return net

        def decode_layers1(input):
            net = UpSampling2D(size=(2,2))(input)
            return net

        def decode_layers2(input):
            net = Conv2D(filters=self.nf*2, kernel_size=1)(input)
            net = _Layer_Depwise_Dncode(input=net, out_channels=self.nf*2)
            net = UpSampling2D(size=(2,2))(net)
            return net

        def decode_layers3(input):
            net = _Layer_Depwise_Dncode(input=input, out_channels=self.nf*2)
            net = UpSampling2D(size=(2,2))(net)
            return net

        def decode_layers4(input):
            net = _Layer_Depwise_Dncode(input=input, out_channels=self.nf*2)
            net = UpSampling2D(size=(2,2))(net)
            return net

        def decode_layers5(input):
            net = _Layer_Depwise_Dncode(input=input, out_channels=self.nf*2)
            net = UpSampling2D(size=(2,2))(net)
            net = _Layer_Depwise_Dncode(input=net, out_channels=self.nf*2)
            net = Conv2D(filters=2, kernel_size=3, activation='sofmax')(net)
            return net

        input = Input(shape=org_shape)
        encode_layer1 = encode_layers1(input)
        encode_layer2 = encode_layers2(encode_layer1)
        encode_layer3 = encode_layers3(encode_layer2)
        encode_layer4 = encode_layers4(encode_layer3)
        encode_layer5 = encode_layers5(encode_layer4)
        decode_layer1 = decode_layers1(encode_layer5)
        decode_layer2 = decode_layers2(decode_layer1)
        decode_layer3 = decode_layers3(decode_layer2)
        decode_layer4 = decode_layers4(decode_layer3)
        out = decode_layers5(decode_layer4)

        super().__init__(input, out)
        self.compile(optimizer='adam', loss=loss)