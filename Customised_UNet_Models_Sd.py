import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Activation, concatenate, Input, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import EfficientNetB7



def conv_block (inputs, num_filters):
  x = Conv2D(num_filters, 3, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(num_filters, 3, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  return x  


def decoder(inputs, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
  x = concatenate([x, skip_features])
  x = conv_block(x, num_filters)
  return x


"""
Unet Model with Efficient_Net_B0 as the Encoder without Attention
"""
def efficent_b0_Unet(input_shape):
  """
    Unet Model with Efficient_Net_B0 as the Encoder without Attention

    args:
        input_shape = Shape of the input Tensors
  """
  
  inputs = Input(input_shape)

  encoder = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)
  encoder.summary()

  # skip connnections
  s1 = encoder.get_layer('input_1').output #256x256x3
  s2 = encoder.get_layer('block1a_project_activation').output #128x128x16
  s3 = encoder.get_layer('block2a_expand_activation').output #64x64x64
  s4 = encoder.get_layer('block4a_expand_activation').output #32x32x192

  # bottle_neck/bridge
  b1 = encoder.get_layer('block6a_expand_activation').output #16x16x672

  # decoder
  d1 = decoder(b1, s4, 512) #32
  d2 = decoder(d1, s3, 256) #64
  d3 = decoder(d2, s2, 128) #128
  d4 = decoder(d3, s1, 64) #256

  #outputs
  outputs = Conv2D(1, 1, padding='same', activation = 'sigmoid')(d4)

  model = Model(inputs, outputs, name='efficent_b0_Unet')
  model.summary()

  return model


"""
Unet Model With DenseNet201 as the Encoder
Without Attention
"""

def unet_Densenet(input_shape):
  """
    Unet Model With DenseNet201 as the Encoder
    Without Attention

    args: 
        input_shape = Shape of the input Tensors
  """
  inputs = Input(input_shape)

  # Encoder Block
  encoder = DenseNet201(include_top=False, weights = 'imagenet', input_tensor = inputs)
  # encoder.summary()

  # Skip Connections
  s1 = encoder.get_layer('input_1').output    #256x256
  s2 = encoder.get_layer('conv1/relu').output     #128x128
  s3 = encoder.get_layer('pool2_relu').output     # 64x64
  s4 = encoder.get_layer('pool3_relu').output     # 32x32

  #bridge
  b1 = encoder.get_layer('pool4_relu').output     #16x16

  # Decoder
  d1 = decoder(b1, s4, 512) # 32x32
  d2 = decoder(d1, s3, 256) # 64x64
  d3 = decoder(d2, s2, 128) # 128x128
  d4 = decoder(d3, s1, 64)  # 256x256

  # Outputs
  outputs = Conv2D(1, 1, padding='same', activation = 'relu')(d4)

  model = Model(inputs, outputs, name='unet_Densenet')
  model.summary()

  return model



"""
Unet Model With DenseNet201 as the Encoder
Without Attention
"""



def efficent_b7_Unet(input_shape):
    """
    Unet Model with Efficient_Net_B7 as the Encoder without Attention

    args:
        input_shape = Shape of the input Tensors
    """
  
    inputs = Input(input_shape)

    encoder = EfficientNetB7(include_top=False, weights='imagenet', input_tensor=inputs)
    encoder.summary()

    # skip connnections
    s1 = encoder.get_layer('input_1').output #256x256x3
    s2 = encoder.get_layer('block2a_expand_activation').output #128x128
    s3 = encoder.get_layer('block3a_expand_activation').output #64x64
    s4 = encoder.get_layer('block4a_expand_activation').output #32x32

    # bottle_neck/bridge
    b1 = encoder.get_layer('block6a_expand_activation').output #16x16x672

    # decoder
    d1 = decoder(b1, s4, 512) #32
    d2 = decoder(d1, s3, 256) #64
    d3 = decoder(d2, s2, 128) #128
    d4 = decoder(d3, s1, 64) #256

    #outputs
    outputs = Conv2D(1, 1, padding='same', activation = 'sigmoid')(d4)

    model = Model(inputs, outputs, name='efficent_b7_Unet')
    model.summary()

    return model



""" Introducing Attention Into the Models """

def attention_gate(inputs, skip_features, num_filters):
#   shape_x = K.int_shape(inputs)
  
  
  gs = Conv2D(num_filters, 1, padding= 'same')(skip_features)
  gs = BatchNormalization()(gs)
#   shape_gs = K.int_shape(gs)


  x = Conv2D(num_filters, 1, padding= 'same')(inputs)
  x = BatchNormalization()(x)

  added_layers = tf.keras.layers.add([x, gs])

  x = Activation('relu')(added_layers)

  x = Conv2D(num_filters, 3, padding= 'same')(x)
  x = BatchNormalization()(x)

  x = Activation('sigmoid')(x)
  # print('inputs_shape', inputs.shape)
  # print('x_shape', x.shape)

  output = tf.keras.layers.multiply([inputs, x])
  

  return output

def decoder_attention_block(inputs, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2, 2), strides = 2, padding = 'same')(inputs)
  attention = attention_gate(x, skip_features, num_filters)
  x = concatenate([x, attention])
  x = conv_block(x, num_filters)
  return x



""" Unet Models along with attention"""

# from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0


## Efficient NetB0
def efficent_b0_attention_Unet(input_shape):
  """ 
  Unet with EfficientNetB0 backbone and attention
  """

  inputs = Input(input_shape)

  #Encoder
  encoder = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=inputs)

  #Skip features
  s1 = encoder.get_layer('input_1').output #256x256x3
  s2 = encoder.get_layer('block1a_project_activation').output #128x128x16
  s3 = encoder.get_layer('block2a_expand_activation').output #64x64x64
  s4 = encoder.get_layer('block4a_expand_activation').output #32x32x192

  # bridge 

  b1 = encoder.get_layer('block6a_expand_activation').output #16x16x672

  # Decoder

  d1 = decoder_attention_block(b1, s4, 512) #32
  d2 = decoder_attention_block(d1, s3, 256) #64
  d3 = decoder_attention_block(d2, s2, 128) #128
  d4 = decoder_attention_block(d3, s1, 64) #256

  #outputs
  outputs = Conv2D(1, 1, padding='same', activation = 'sigmoid')(d4)

  model = Model(inputs, outputs, name='efficent_b0_Unet')
  model.summary()

  return model


## Densenet201

def unet_attention_Densenet(input_shape):
  inputs = Input(input_shape)

  # Encoder Block
  encoder = DenseNet201(include_top=False, weights = 'imagenet', input_tensor = inputs)
  # encoder.summary()

  # Skip Connections
  s1 = encoder.get_layer('input_1').output    #256x256
  s2 = encoder.get_layer('conv1/relu').output     #128x128
  s3 = encoder.get_layer('pool2_relu').output     # 64x64
  s4 = encoder.get_layer('pool3_relu').output     # 32x32

  #bridge
  b1 = encoder.get_layer('pool4_relu').output     #16x16

  # Decoder
  d1 = decoder_attention_block(b1, s4, 512) # 32x32
  d2 = decoder_attention_block(d1, s3, 256) # 64x64
  d3 = decoder_attention_block(d2, s2, 128) # 128x128
  d4 = decoder_attention_block(d3, s1, 64)  # 256x256

  # Outputs
  outputs = Conv2D(1, 1, padding='same', activation = 'relu')(d4)

  model = Model(inputs, outputs, name='unet_Densenet')
  model.summary()

  return model





def efficent_b7_attention_Unet(input_shape):
  
  inputs = Input(input_shape)

  encoder = EfficientNetB7(include_top=False, weights='imagenet', input_tensor=inputs)
  encoder.summary()

  # skip connnections
  s1 = encoder.get_layer('input_1').output #256x256x3
  s2 = encoder.get_layer('block2a_expand_activation').output #128x128
  s3 = encoder.get_layer('block3a_expand_activation').output #64x64
  s4 = encoder.get_layer('block4a_expand_activation').output #32x32

  # bottle_neck/bridge
  b1 = encoder.get_layer('block6a_expand_activation').output #16x16x672

  # decoder
  d1 = decoder_attention_block(b1, s4, 512) #32
  d2 = decoder_attention_block(d1, s3, 256) #64
  d3 = decoder_attention_block(d2, s2, 128) #128
  d4 = decoder_attention_block(d3, s1, 64) #256

  #outputs
  outputs = Conv2D(1, 1, padding='same', activation = 'sigmoid')(d4)

  model = Model(inputs, outputs, name='efficent_b7_Unet')
  model.summary()

  return model
