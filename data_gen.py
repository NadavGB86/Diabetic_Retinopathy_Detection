from keras.preprocessing.image import ImageDataGenerator
import get_dataset


def get_data_generators(train_df, valid_df, test_df, image_size=128, BS=32):

    trainAug = ImageDataGenerator(
        rescale=1 / 255.0,
        # featurewise_center=True, samplewise_center=True,
        # featurewise_std_normalization=True, samplewise_std_normalization=True,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")

    valAug = ImageDataGenerator(rescale=1 / 255.0)

    trainGen = trainAug.flow_from_dataframe(
        train_df,
        x_col='File Path', y_col='Label',
        class_mode="categorical",
        target_size=(image_size, image_size),
        color_mode="rgb",
        shuffle=True,
        batch_size=BS)

    valGen = valAug.flow_from_dataframe(
        valid_df,
        x_col='File Path', y_col='Label',
        class_mode="categorical",
        target_size=(image_size, image_size),
        color_mode="rgb",
        shuffle=False,
        batch_size=BS)

    testGen = valAug.flow_from_dataframe(
        test_df,
        x_col='File Path', y_col='Label',
        class_mode="categorical",
        target_size=(image_size, image_size),
        color_mode="rgb",
        shuffle=False,
        batch_size=BS)

    return trainGen, valGen, testGen







