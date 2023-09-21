import tensorflow.keras as ks
from keras.layers import Input, SeparableConv2D, BatchNormalization, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate, Conv2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.optimizers import Adam
from custom_metrics import dice_coef_multilabel


def build_cnn(input_shape, n_ch, L=3):
    x = ks.layers.Input(shape=input_shape)

    h = x
    for i in range(L):
        h = ks.layers.Conv2D(n_ch, kernel_size=3, padding='same')(h)
        h = ks.layers.ReLU()(h)

        n_ch = 2*n_ch

    y = ks.layers.Conv2D(1, kernel_size=1, activation='sigmoid')(h)
    return ks.models.Model(x, y)


def get_model(img_size, num_classes):
    inputs = ks.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = ks.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = ks.layers.BatchNormalization()(x)
    x = ks.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = ks.layers.Activation("relu")(x)
        x = ks.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = ks.layers.BatchNormalization()(x)

        x = ks.layers.Activation("relu")(x)
        x = ks.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = ks.layers.BatchNormalization()(x)

        x = ks.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = ks.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = ks.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = ks.layers.Activation("relu")(x)
        x = ks.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = ks.layers.BatchNormalization()(x)

        x = ks.layers.Activation("relu")(x)
        x = ks.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = ks.layers.BatchNormalization()(x)

        x = ks.layers.UpSampling2D(2)(x)

        # Project residual
        residual = ks.layers.UpSampling2D(2)(previous_block_activation)
        residual = ks.layers.Conv2D(filters, 1, padding="same")(residual)
        x = ks.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = ks.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(
        x
    )

    # Define the model
    model = ks.Model(inputs, outputs)
    return model


# Taking a batch of test inputs to measure model's progress.
# test_images, test_masks = next(iter(resized_val_ds))
def depthwise_pointwise_conv(in_ch, out_ch, kernel_size, padding, channels_per_seg=128):
    C = 3 if in_ch == 3 and channels_per_seg != 1 else min(in_ch, channels_per_seg)

    out_conv = ks.Sequential()
    out_conv.add(ks.layers.conv2d(in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch // C))
    out_conv.add(ks.layers.conv2d(in_ch, kernel_size=1, padding=0, groups=1))

    # TODO check what is out_channel and use that properly

    # self.conv = nn.Sequential(
    #     nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch // C),
    #     nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, groups=1),
    #
    # )
    return out_conv


def double_conv(in_channels, out_channels, mid_channels=None):
    if not mid_channels:
        mid_channels = out_channels


    double_conv = ks.Sequential()
    double_conv.add(depthwise_pointwise_conv(in_channels, mid_channels, kernel_size=3, padding=1))
    double_conv.add(ks.layers.BatchNormalization()) # TODO check if this is the right way to do it
    double_conv.add(ks.layers.ReLU(inplace=True))
    double_conv.add(depthwise_pointwise_conv(mid_channels, out_channels, kernel_size=3, padding=1))
    double_conv.add(ks.layers.BatchNormalization())
    double_conv.add(ks.layers.ReLU(inplace=True))

    # self.double_conv = nn.Sequential(
    #     depthwise_pointwise_conv(in_channels, mid_channels, kernel_size=3, padding=1),
    #     nn.BatchNorm2d(mid_channels),
    #     nn.ReLU(inplace=True),
    #     depthwise_pointwise_conv(mid_channels, out_channels, kernel_size=3, padding=1),
    #     nn.BatchNorm2d(out_channels),
    #     nn.ReLU(inplace=True)
    # )
    return double_conv


def down(in_channels, out_channels):

    down_layer = ks.Sequential()
    down_layer.add(ks.layers.MaxPool2D(2))
    down_layer.add(double_conv(in_channels, out_channels))
    # self.maxpool_conv = nn.Sequential(
    #     nn.MaxPool2d(2),
    #     DoubleConv(in_channels, out_channels)
    # )
    return down_layer


def up(self, in_channels, out_channels, bilinear=True):
    # if bilinear, use the normal convolutions to reduce the number of channels

    up_layer = ks.Sequential()
    if bilinear:
        up_layer.add(ks.layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        up_layer.add(double_conv(in_channels, out_channels, in_channels // 2))
    else:
        up_layer.add(ks.layers.Conv2DTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2))
        up_layer.add(double_conv(in_channels, out_channels))

    # if bilinear:
    #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    #     self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    # else:
    #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    #     self.conv = DoubleConv(in_channels, out_channels)
    return up_layer


def out_conv(in_channels, out_channels):
    # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    return ks.layers.Conv2D(in_channels, kernel_size=1) # TODO check if this is the right way to do it


def get_lighter_unet(n_channels, n_classes, bilinear=True):
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear


    self.inc = DoubleConv(n_channels, 64)
    self.down1 = Down(64, 128)
    self.down2 = Down(128, 256)
    self.down3 = Down(256, 512)
    factor = 2 if bilinear else 1
    self.down4 = Down(512, 1024 // factor)
    self.up1 = Up(1024, 512 // factor, bilinear)
    self.up2 = Up(512, 256 // factor, bilinear)
    self.up3 = Up(256, 128 // factor, bilinear)
    self.up4 = Up(128, 64, bilinear)
    self.outc = OutConv(64, n_classes)

    self.dropout = nn.Dropout(0.5)  # dropout
    pass


def mobileunet(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)

    conv1 = SeparableConv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = SeparableConv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = SeparableConv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = SeparableConv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = SeparableConv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = SeparableConv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = SeparableConv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = SeparableConv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = SeparableConv2D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    # conv6 = Conv2DTranspose(512, 3, strides=(2, 2), activation='relu', padding='same')(conv5)
    conv6 = Conv2DTranspose(512, 3, strides=(2, 2), activation='relu', padding='same')(conv5)
    cat6 = concatenate([conv4, conv6], axis=3)
    conv6 = SeparableConv2D(512, 3, activation='relu', padding='same')(cat6)
    conv6 = BatchNormalization()(conv6)
    conv6 = SeparableConv2D(512, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    conv7 = Conv2DTranspose(256, 3, strides=(2, 2), activation='relu', padding='same')(conv6)
    cat7 = concatenate([conv3, conv7], axis=3)
    conv7 = SeparableConv2D(256, 3, activation='relu', padding='same')(cat7)
    conv7 = BatchNormalization()(conv7)
    conv7 = SeparableConv2D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv2DTranspose(128, 3, strides=(2, 2), activation='relu', padding='same')(conv7)
    cat8 = concatenate([conv2, conv8], axis=3)
    conv8 = SeparableConv2D(128, 3, activation='relu', padding='same')(cat8)
    conv8 = BatchNormalization()(conv8)
    conv8 = SeparableConv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv2DTranspose(64, 3, strides=(2, 2), activation='relu', padding='same')(conv8)
    cat9 = concatenate([conv1, conv9], axis=3)
    conv9 = SeparableConv2D(64, 3, activation='relu', padding='same')(cat9)
    conv9 = BatchNormalization()(conv9)
    conv9 = SeparableConv2D(64, 3, activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(8, 1, activation='softmax')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy',
                  metrics=[dice_coef_multilabel], run_eagerly=True)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def mod_mobileunet(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)

    conv1 = SeparableConv2D(64, 3, activation='elu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(64, 3, activation='elu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = SeparableConv2D(128, 3, activation='elu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = SeparableConv2D(128, 3, activation='elu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = SeparableConv2D(256, 3, activation='elu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = SeparableConv2D(256, 3, activation='elu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)

    conv8 = Conv2DTranspose(128, 3, strides=(2, 2), activation='elu', padding='same')(conv3)
    cat8 = concatenate([conv2, conv8], axis=3)
    conv8 = SeparableConv2D(128, 3, activation='elu', padding='same')(cat8)
    conv8 = BatchNormalization()(conv8)
    conv8 = SeparableConv2D(128, 3, activation='elu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    conv9 = Conv2DTranspose(64, 3, strides=(2, 2), activation='elu', padding='same')(conv8)
    cat9 = concatenate([conv1, conv9], axis=3)
    conv9 = SeparableConv2D(64, 3, activation='elu', padding='same')(cat9)
    conv9 = BatchNormalization()(conv9)
    conv9 = SeparableConv2D(64, 3, activation='elu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(7, 1, activation='softmax')(conv9)

    model = Model(inputs, conv10)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy',
                  metrics=[dice_coef_multilabel], run_eagerly=True)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


