import pandas as pd
import tensorflow as tf


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanSquaredError())
    return model


data_path = r"C:\Users\Klomm\OneDrive\Desktop\AI_SERIES\Learn\2021-07-06\resources\abalone.csv"
cols_name = ["length", "diameter", "height", "whole weight",
             "shucked weight", "viscere weight", "shell weight", "age"]
df = pd.read_csv(data_path, names=cols_name)
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values
model = create_model()
model.fit(features, labels, epochs=100)

