import tensorflow as tf
import matplotlib.pyplot as plt
import os


def plot_img(img):
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        fig.add_subplot(5, 5, i+1)
        plt.xticks()
        plt.yticks()
        plt.imshow(img[i])
    plt.show()


def crete_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def main():
    dataset = tf.keras.datasets.cifar10.load_data()
    model_name = "cifar10.mdl"
    output_path = f"{os.getcwd()}/models/{model_name}"
    (X_train, y_train), (X_test, y_test) = dataset
    # print(X_train.shape)
    plot_img(X_train[50000-25:50000])
    model = crete_model()

    print(model.summary())

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=128)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"LOSS: {test_loss}")
    print(f"Accuracy {test_accuracy}")
    model.save(output_path)


main()
