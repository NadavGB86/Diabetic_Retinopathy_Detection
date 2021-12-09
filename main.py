# import ctypes
#
# ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cudart64_110.dll")
# ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cublas64_11")
# ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cufft64_10.dll")
# ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\curand64_10.dll")
# ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cusolver64_11.dll")
# ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cusparse64_11.dll")
# ctypes.WinDLL("C:\\tools\\cuda\\bin\\cudnn64_8.dll")

import tensorflow as tf
import tensorflow_addons as tfa
import data_gen
import model
import get_dataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    NUM_EPOCHS = 40
    IMAGE_SIZE = 128
    BS = 32

    train_df, valid_df, test_df = get_dataset.get_datasets()
    trainGen, valGen, testGen = data_gen.get_data_generators(train_df, valid_df, test_df, image_size=IMAGE_SIZE, BS=32)
    model = model.get_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # by default learning_rate=0.001
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="cat_acc"), tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision'),
                 tfa.metrics.CohenKappa(num_classes=5, sparse_labels=False, weightage='quadratic')]
    )

    history = model.fit(
        trainGen,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_df) // BS,
        validation_data=valGen,
        validation_steps=len(valid_df) // BS,
        class_weight={0: len(train_df[train_df['Label'] == '0']) / len(train_df),
                      1: len(train_df[train_df['Label'] == '1']) / len(train_df),
                      2: len(train_df[train_df['Label'] == '2']) / len(train_df),
                      3: len(train_df[train_df['Label'] == '3']) / len(train_df),
                      4: len(train_df[train_df['Label'] == '4']) / len(train_df)},
        shuffle=True,
        callbacks=[
            # tf.keras.callbacks.EarlyStopping(patience=11, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath='bestmodel.h5', save_best_only=True, verbose=1)]
    )
