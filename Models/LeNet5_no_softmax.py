from tensorflow.keras import Sequential, layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def LeNet(n_classes=10, input_shape=(32, 32, 1)):
    """ Build a LeNet 5 Model """
    model = Sequential()

    # First convolution block: (*, 32, 32, 1) -Conv2D+ReLU-> (*, 28, 28, 6) -MaxPooling2D-> (*, 14, 14, 6)
    # 1) Convolution layer: 6 channels, 5*5 kernel, ReLU activation, padded to maintain same shape
    # 2) Average pooling layer: 2*2 kernel
    # NOTE: If image inputted is not (28, 28, 1), then add padding="same" in 1st convolution layer
    padding='valid'
    if (input_shape==(28,28,1)):
        padding='same'
    model.add(layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=input_shape, strides=(1,1), padding=padding))
    model.add(layers.AveragePooling2D((2, 2)))

    # Second convolution block: (*, 14, 14, 6) -Conv2D+ReLU-> (*, 10, 10, 16) -MaxPooling2D-> (*, 5, 5, 16)
    # (Parameters have same meanings as the first block)
    model.add(layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu', padding='valid'))
    model.add(layers.AveragePooling2D((2, 2)))

    # Third convolution block: (*, 5, 5, 16) -Conv2D+ReLU-> (*, 1, 1, 120)
    model.add(layers.Conv2D(filters=120, kernel_size=(5,5), activation='relu', padding='valid'))

    # Flatten layer reshape features to 1D: (*, 1, 1, 120) -Flatten-> (*, 120)
    model.add(layers.Flatten())

    # First fully-connected (linear) layer: (*, 1250) -FC+ReLU -> (*, 84)
    model.add(layers.Dense(84, activation="relu"))

    # Last fully-connected (linear) layer: (*, 500) -FC+Softmax-> (*, n_classes)
    # Softmax outputs a categorical distribution representing probability for each character
    model.add(layers.Dense(n_classes))

    # Build classification model with SGD optimizer and cross entropy loss
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer="SGD", metrics=['accuracy'])

    return model