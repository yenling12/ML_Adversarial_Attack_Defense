#############################################################
## VGG16 Model (Pretrained)
# Transfer Learning: Pre-trained VGG-16 Model
# Reference: https://www.kaggle.com/code/viratkothari/image-classification-of-mnist-using-vgg16#5.-Building-a-Model:-Using-Transfer-Learning
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def VGG16_pretrained(n_classes=10, input_shape=(48,48,3)):
    """ Build a VGG_16 pretrained model. NOTE: Input size minimum is (32,32,3)"""
    
    # Initiale original VGG-16 model with weights='imagenet'.
    model_vgg16_orig=VGG16(weights='imagenet')
    # Print VGG-16 original architecture summary
    #model_vgg16_orig.summary()

    ############################
    # Now we need to edit the original VGG-16 to fit our purpose.
    # Prepare our input_layer to pass our image size. Input will be input_shape.
    input_layer=layers.Input(shape=input_shape)

    # Initialize the transfer model VGG16 with appropriate properties for MNIST data
    # we are passing paramers as following
    # 1) weights='imagenet' - Using this we are carring weights as of original weights.
    # 2) input_tensor is used to change our input layer expected size
    # 3) we want to change the last layer so add include_top=False-> this will remove the
    #    the below layers:
    #           block5_pool (MaxPooling2D) (None, 7, 7, 512) 0
    #           flatten (Flatten) (None, 25088) 0
    #           fc1 (Dense) (None, 4096) 102764544
    #           fc2 (Dense) (None, 4096) 16781312
    #           predictions (Dense) (None, 1000) 4097000
    model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)

    # Next we take the last layer and....
    last_layer=model_vgg16.output 

    # flatten the last layer
    flatten=layers.Flatten()(last_layer) 

    # then add the Dense layer to the flattened layer
    output_layer=layers.Dense(n_classes)(flatten)

    # Then create the VGG16 model with the new output layer
    model_vgg16=models.Model(inputs=model_vgg16.input, outputs=output_layer)
    
    # Then we making all the layers intrainable except the last layer
    print("NOTE: All the layers are intrainable except the last layer. \n")
    for layer in model_vgg16.layers[:-1]:
        layer.trainable=False
    
    # Finally add classification model 
    model_vgg16.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',metrics=['accuracy'])
    
    return model_vgg16