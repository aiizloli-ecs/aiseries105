import os
import tensorflow as tf
import matplotlib.pyplot as plt


def create_plot(ds, data_aug=None):
    fig = plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            fig.add_subplot(3, 3, i + 1)
            if data_aug is None:
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.title(int(labels[i]))
            else:
                aug_img = data_aug(images)
                plt.imshow(aug_img[0].numpy().astype('uint8'))
            plt.axis('off')
    plt.show()


def make_model(input_shape, num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)

    x = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)


def make_dataset(dataset_path, val_split=0.2, img_size=(64, 64), batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=val_split,
        subset="training",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=val_split,
        subset="validation",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size
    )
    return train_ds, val_ds


def main():
    # set parameter
    dataset_path = r"C:\Users\Klomm\OneDrive\Desktop\AI_SERIES\Learn\2021-07-06\dataset\gender"
    model_name = f"{os.path.basename(dataset_path)}.h5"
    num_class = len(os.listdir(dataset_path))
    DATA = {"DATA_PATH": dataset_path,
            "NUM_CLASS": num_class,
            "VAL_SPLIT": 0.2,
            "BATCH_SIZE": 64,
            "IMG_SIZE": (128, 128),
            "EPOCHS": 10,
            "MODEL_NAME": model_name,
            "CHECKPOINT_MODEL": f"{os.getcwd()}/checkpoints/{model_name}"
            }

    # make dataset
    train_ds, val_ds = make_dataset(DATA["DATA_PATH"],
                                    val_split=DATA["VAL_SPLIT"],
                                    img_size=DATA["IMG_SIZE"],
                                    batch_size=DATA["BATCH_SIZE"])

    # visualize dataset
    create_plot(train_ds)

    # make model
    model = make_model(input_shape=DATA["IMG_SIZE"] + (3, ),
                       num_classes=DATA["NUM_CLASS"])
    CALLBACKS = [
        tf.keras.callbacks.ModelCheckpoint(DATA["CHECKPOINT_MODEL"])
    ]
    loss = tf.keras.losses.binary_crossentropy if num_class == 2 \
        else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=loss,
                  metrics=["accuracy"])

    # visualize model
    print(model.summary())

    # train model
    model.fit(
        train_ds, epochs=DATA["EPOCHS"], callbacks=CALLBACKS, validation_data=val_ds
    )


# run main program
main()
