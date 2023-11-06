from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout,Conv2DTranspose, BatchNormalization, Add, Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

smooth = 1.
#smooth = 10e-5

####################
## Loss Functions ##
####################
# Dice coefficient
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

# Dice loss
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

# Dual dice coefficient
def dual_dice_coef(y_true, y_pred):
    Foreground = dice_coef(y_true, y_pred)
    Background = dice_coef(1. - y_true, 1. - y_pred)
    return 0.5 * (Foreground + Background)

# Dual dice loss
def dual_dice_coef_loss(y_true, y_pred):
    return 1.0 - dual_dice_coef(y_true, y_pred)

####################
## Resunet Blocks ##
####################

# Pooling layer
def pooling_step(x,kernel_size=(2,2)):
    pool1 = MaxPooling2D(kernel_size)(x)
    return pool1

# First step
def first_step(x, num_filters,bt_norm=True, kernel_size=(3, 3),act="relu", pad="same", stride=1):
    conv1 = Conv2D(num_filters, kernel_size, padding=pad,strides=stride)(x) # Convolution
    if bt_norm == True:
        conv1 = BatchNormalization()(conv1) # Batch normalization
    conv1 = Activation(act)(conv1) # ReLU Activation
    conv1 = Conv2D(num_filters, kernel_size, padding=pad,strides=1)(conv1)

    shortcut = Conv2D(num_filters, kernel_size=(1, 1), padding=pad, strides=stride)(x)
    shortcut = BatchNormalization()(shortcut)

    final = Add()([conv1,shortcut]) 
    return final

# Convolution block
def res_conv_block(x, num_filters,bt_norm=True, kernel_size=(3, 3),act="relu", pad="same", stride=1):
    if bt_norm == True:
        x1 = BatchNormalization()(x)
    else:
        x1 = x
    conv1 = Activation(act)(x1)
    conv1 = Conv2D(num_filters, kernel_size, padding=pad, strides=stride)(conv1)
    if bt_norm == True:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation(act)(conv1)
    conv1 = Conv2D(num_filters, kernel_size, padding=pad, strides=1)(conv1)

    shortcut = Conv2D(num_filters, kernel_size=(1, 1), padding=pad, strides=stride)(x)
    shortcut = BatchNormalization()(shortcut)

    final = Add()([conv1,shortcut])
    return final

# Decoder block
def res_decoder_step(x, encoder_layer, num_filters,bt_norm=True, kernel_size=(3, 3),act="relu", pad="same", stride=1):
    deconv4 = Conv2DTranspose(num_filters, kernel_size, activation=act, padding=pad, strides=(2, 2))(x)
    if bt_norm == True:
        deconv4 = BatchNormalization()(deconv4)
    uconv4 = concatenate([deconv4, encoder_layer])
    uconv4 = res_conv_block(uconv4,num_filters,bt_norm, kernel_size, act, pad, stride)
    return uconv4
    
##################################################
## Central blocks to encode global information ###
##################################################

# Atrous Convolution block
def atrous_conv(x, num_filters, bt_norm=True):
    inputs_size = tf.shape(x)[1:3]
    conv1 = Conv2D(num_filters, (1,1), padding='same',strides=1, dilation_rate=1)(x)
    conv2 = Conv2D(num_filters, (3,3), padding='same',strides=1, dilation_rate=6)(x)
    conv3 = Conv2D(num_filters, (3,3), padding='same',strides=1, dilation_rate=12)(x)
    conv4 = Conv2D(num_filters, (3,3), padding='same',strides=1, dilation_rate=18)(x)

    image_level_features = tf.reduce_mean(x, [1, 2], keepdims=True, name='global_average_pooling')
    # 1x1 convolution with 256 filters( and batch normalization)
    image_level_features = Conv2D(num_filters, (1, 1), strides=1)(image_level_features)
    # bilinearly upsample features
    image_level_features = tf.image.resize(image_level_features, inputs_size, name='upsample')

    net = tf.concat([conv1, conv2, conv3, conv4, image_level_features], axis=3, name='concat')
    net = Conv2D(num_filters, (1, 1), strides=1)(net)

    return net
    
## Transformer layers
# MLP Layer
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

# Create Patches from input image
class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Patch Encoder
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# Transformer
def transformer_mid(input, num_filters, image_size = 32, patch_size = 1, projection_dim = 128, num_heads = 4, transformer_layers = 8, mlp_head_units = [2048, 1024], bt_norm=True):
    print(input.shape)
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [
    projection_dim * 2,
    projection_dim,
    ]  # Size of the transformer layers
    # Create patches.
    patches = Patches(patch_size)(input)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    print(encoded_patches.shape)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    print(representation.shape)

######################
## Different models ##
######################

# ResUnet with pooling layer
def resunet_with_pool(start_neurons,img_rows,img_cols,bt_state=True):
    strategy = tf.distribute.MirroredStrategy() # Train on multiple GPUs
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # Train across multiple nodes
    with strategy.scope():
        input_layer = Input((img_rows,img_cols, 1))
        if bt_state == True:
            input_layer1 = BatchNormalization()(input_layer)
        else:
            input_layer1 = input_layer
        
        # Encoder
        conv1 = first_step(input_layer1,start_neurons * 1, bt_norm=bt_state)
        pool1 = pooling_step(conv1, kernel_size=(2, 2))
        conv2 = res_conv_block(pool1,start_neurons * 2, bt_norm=bt_state)
        pool2 = pooling_step(conv2, kernel_size=(2, 2))
        conv3 = res_conv_block(pool2, start_neurons * 4, bt_norm=bt_state)
        pool3 = pooling_step(conv3, kernel_size=(2, 2))
        conv4 = res_conv_block(pool3, start_neurons * 8, bt_norm=bt_state)
        pool4 = pooling_step(conv4, kernel_size=(2, 2))

        # Middle
        conv5 = res_conv_block(pool4, start_neurons * 16, bt_norm=bt_state)

        # Decoder
        deconv4 = res_decoder_step(conv5, conv4, start_neurons * 8, bt_norm=bt_state)
        deconv3 = res_decoder_step(deconv4, conv3, start_neurons * 4, bt_norm=bt_state)
        deconv2 = res_decoder_step(deconv3, conv2, start_neurons * 2, bt_norm=bt_state)
        deconv1 = res_decoder_step(deconv2, conv1, start_neurons * 1, bt_norm=bt_state)

        output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(deconv1)

        model = Model(input_layer,output_layer)

        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model

# ResUnet without pooling layer (with three center possibilities: 1. Normal ResUNet, 2. Atrous block, 3. Transformer Block)
def resunet_without_pool(start_neurons, img_rows, img_cols, learning_rate, bt_state=True):
    strategy = tf.distribute.MirroredStrategy() # Train on multiple GPUs
    #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # Train across multiple nodes
    with strategy.scope():
        input_layer = Input((img_rows,img_cols, 1))
        if bt_state == True:
            input_layer1 = BatchNormalization()(input_layer)
        else:
            input_layer1 = input_layer

        # Encoder
        conv1 = first_step(input_layer1,start_neurons * 1, bt_norm=bt_state)
        conv2 = res_conv_block(conv1,start_neurons * 2, stride=2, bt_norm=bt_state)
        conv3 = res_conv_block(conv2, start_neurons * 4, stride=2, bt_norm=bt_state)
        conv4 = res_conv_block(conv3, start_neurons * 8, stride=2,  bt_norm=bt_state)
        conv5 = res_conv_block(conv4, start_neurons * 8, stride=2, bt_norm=bt_state)

        # Middle
        conv6 = res_conv_block(conv5, start_neurons * 16, bt_norm=bt_state)
        if center == 'atrous':
            conv6 = atrous_conv(conv6, num_filters=256, bt_norm=True) # Atrous convolution block
        elif center == 'trans':
            conv6 = transformer_mid(conv6, num_filters=256, bt_norm=True) # Transformer block
        
        
        # Decoder
        deconv4 = res_decoder_step(conv6, conv4, start_neurons * 8, bt_norm=bt_state)
        deconv3 = res_decoder_step(deconv4, conv3, start_neurons * 4, bt_norm=bt_state)
        deconv2 = res_decoder_step(deconv3, conv2, start_neurons * 2, bt_norm=bt_state)
        deconv1 = res_decoder_step(deconv2, conv1, start_neurons * 1, bt_norm=bt_state)

        output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(deconv1)

        model = Model(input_layer,output_layer)

        model.compile(optimizer=Adam(lr = learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model

if __name__ == "__main__":
    """
    testing
    """
    model = resunet_with_pool(64, 512, 512,bt_state=True)
    model.summary()
