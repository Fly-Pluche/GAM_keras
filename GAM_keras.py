import tensorflow as tf
from keras import layers,Sequential
from keras.layers import Conv2D
from keras.layers import BatchNormalization, Dropout, Dense
from keras.layers import ReLU

class GAM(layers.Layer):
    def __init__(self, in_channels, out_channels, rate=4):
        super().__init__()
        inchannel_rate = int(int(in_channels)/rate)

        self.channel_attention = Sequential()

        self.channel_attention.add(Dense(inchannel_rate))
        self.channel_attention.add(ReLU())
        self.channel_attention.add(Dense(in_channels))

        
        self.spatial_attention = Sequential()

        self.spatial_attention.add(Conv2D(inchannel_rate,kernel_size=(7,7),padding='same'))
        self.spatial_attention.add(BatchNormalization())
        self.spatial_attention.add(ReLU())
        self.spatial_attention.add(Conv2D(out_channels,kernel_size=(7,7),padding='same'))
        self.spatial_attention.add(BatchNormalization())

    def forward(self,x):
        b, c, h, w = x.shape
        # B,C,H,W ==> B,H*W,C
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        
        # B,H*W,C ==> B,H,W,C
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        # B,H,W,C ==> B,C,H,W
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

if __name__ == '__main__':
    img = tf.random.normal([1,64,32,48])
    b, c, h, w = img.shape
    net = GAM(in_channels=c, out_channels=c)
    output = net(img)
    print(output.shape)