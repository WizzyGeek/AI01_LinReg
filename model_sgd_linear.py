import tensorflow as tf
import numpy as np

fd = open("challenge_data.npy", "rb")
arr = np.load(fd)

# print(tf.config.list_physical_devices())
print(arr[0], arr[1])

model = tf.keras.Sequential(
    [
        tf.keras.Input(1),
        tf.keras.layers.Dense(1)
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss=tf.keras.losses.MeanSquaredError())
model.fit(arr[0], arr[1], epochs=100, validation_split=0.2)
# print(model.get_weight_paths())
print(model.weights)

import matplotlib.pyplot as plt

plt.scatter(arr[0], arr[1])
plt.plot((x:=tf.linspace(0.0, 100, 100)), model.predict(x))
plt.show()
