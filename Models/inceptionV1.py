from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform, Constant
from tensorflow.keras import layers

# Reference: https://nitishkumarpilla.medium.com/understand-googlenet-inception-v1-and-implement-it-easily-from-scratch-using-tensorflow-and-keras-5404239f361

def inceptionV1(n_classes=10, input_shape=(224, 224, 3)):
    '''Build an InceptionV1 Model'''
    # Initializers - Both weight and bias initializers are tools to 
    #  help the neural network learn more effectively during training.
    kernel_init = glorot_uniform() # weight initializer-helps with convergence speed
    bias_init = Constant(value=0.2) # bias initializer-helps with learning complex patterns

    # Inception Module
    def inception_module(x,filters_1x1,filters_3x3_reduce,filters_3x3,filters_5x5_reduce,filters_5x5,filters_pool_proj,kernel_init,bias_init, name=None):
        conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)
        conv_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)
        pool_proj = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)
        output = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
        return output
    
    # Building the Incpetion V1 Model Architecture
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)
    x = inception_module(x,filters_1x1=64,filters_3x3_reduce=96, filters_3x3=128,filters_5x5_reduce=16, filters_5x5=32,filters_pool_proj=32, kernel_init=kernel_init, bias_init=bias_init, name='inception_3a')
    x = inception_module(x,filters_1x1=128,filters_3x3_reduce=128, filters_3x3=192,filters_5x5_reduce=32, filters_5x5=96,filters_pool_proj=64, kernel_init=kernel_init, bias_init=bias_init,name='inception_3b')
    x = layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    x = inception_module(x,filters_1x1=192,filters_3x3_reduce=96, filters_3x3=208,filters_5x5_reduce=16,filters_5x5=48,filters_pool_proj=64, kernel_init=kernel_init, bias_init=bias_init,name='inception_4a')
    x1 = layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dense(1024, activation='relu')(x1)
    x1 = layers.Dropout(0.7)(x1)
    x1 = layers.Dense(10, activation='softmax', name='auxilliary_output_1')(x1)
    x = inception_module(x,filters_1x1=160,filters_3x3_reduce=112, filters_3x3=224,filters_5x5_reduce=24,filters_5x5=64,filters_pool_proj=64, kernel_init=kernel_init, bias_init=bias_init,name='inception_4b')
    x = inception_module(x,filters_1x1=128,filters_3x3_reduce=128, filters_3x3=256,filters_5x5_reduce=24,filters_5x5=64, filters_pool_proj=64, kernel_init=kernel_init, bias_init=bias_init,name='inception_4c')
    x = inception_module(x,filters_1x1=112,filters_3x3_reduce=144, filters_3x3=288,filters_5x5_reduce=32,filters_5x5=64, filters_pool_proj=64, kernel_init=kernel_init, bias_init=bias_init, name='inception_4d')
    x2 = layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(1024, activation='relu')(x2)
    x2 = layers.Dropout(0.7)(x2)
    x2 = layers.Dense(10, activation='softmax', name='auxilliary_output_2')(x2)
    x = inception_module(x,filters_1x1=256,filters_3x3_reduce=160, filters_3x3=320,filters_5x5_reduce=32,filters_5x5=128, filters_pool_proj=128, kernel_init=kernel_init, bias_init=bias_init, name='inception_4e')
    x = layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    x = inception_module(x,filters_1x1=256,filters_3x3_reduce=160, filters_3x3=320,filters_5x5_reduce=32,filters_5x5=128, filters_pool_proj=128, kernel_init=kernel_init, bias_init=bias_init, name='inception_5a')
    x = inception_module(x, filters_1x1=384,filters_3x3_reduce=192, filters_3x3=384,filters_5x5_reduce=48,filters_5x5=128, filters_pool_proj=128, kernel_init=kernel_init, bias_init=bias_init, name='inception_5b')
    x = layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = Model(input_layer, [x, x1, x2], name='inception_v1')

    # Compiling the Model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
