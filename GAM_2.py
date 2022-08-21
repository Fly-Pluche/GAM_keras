import tensorflow as tf
from keras import layers

from keras.layers import Conv2D
from keras.layers import BatchNormalization, Dense
from keras.activations import relu


#tensorflow             1.4.0
#Keras                  2.0.8

class GAM(layers.Layer):
    def __init__(self, in_channels, out_channels, rate=4):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels/rate)


        self.dense1 = Dense(inchannel_rate,input_shape=(in_channels,),activation='relu')

        self.dense2 = Dense(in_channels)
        

        self.conv1=Conv2D(inchannel_rate,kernel_size=(7,7),padding='same')

        self.conv2=Conv2D(out_channels,kernel_size=(7,7),padding='same')


    def forward(self,x):
        b, c, h, w = x.shape
        # B,C,H,W ==> B,H*W,C
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        
        # B,H*W,C ==> B,H,W,C
        print('x_permute',x_permute.shape)
        x_att_permute = self.dense2(self.dense1(x_permute)).view(b, h, w, c)
        # B,H,W,C ==> B,C,H,W
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = relu(BatchNormalization(self.conv1(x)))
        x_spatial_att = BatchNormalization(self.conv2(x)).sigmoid()
        
        out = x * x_spatial_att

        return out

if __name__ == '__main__':
    img = tf.random_normal([1,64,32,48])
    b, c, h, w = img.shape
    net = GAM(in_channels=c, out_channels=c)
    output = net(img)
    print(output.shape)



