import tensorflow as tf
import matplotlib.pyplot as plt


def plot_img(img, row, col):
    fig, axs = plt.subplots(row, col)
    num_img = 0
    for row in range(row):
        for col in range(col):
            axs[row, col].imshow(img[num_img])
            num_img += 1
    plt.show()


def crete_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def main():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = crete_model()
    print(X_train.shape, y_train.shape)
    model.fit(X_train, y_train, epochs=1000)
    eval_ = model.evaluate(X_test, y_test, verbose=2)
    print(eval_)


main()
