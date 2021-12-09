from tensorflow.keras.layers import Input  # Input Layer
from tensorflow.keras.applications import DenseNet121  # Keras Application
from tensorflow.keras.layers import Dense  # Dense Layer (Fully connected)
from tensorflow.keras.models import Model  # Model Structure


def get_model(image_size=128):
    input_shape = (image_size, image_size, 3)
    img_input = Input(shape=input_shape)
    base_model = DenseNet121(include_top=False,
                             input_tensor=img_input,
                             input_shape=input_shape,
                             pooling="max",
                             weights='imagenet')
    base_model.trainable = True
    x = base_model.output
    predictions = Dense(5, activation="softmax", name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)
    return model
