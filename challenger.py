import numpy as np

slope = 0.269
yinter = 1.35

n = 100

error = np.random.normal(scale=0.8, size=n)

x_val = np.arange(n)
y_val = x_val * slope + yinter

y_val_error = y_val + error

arr = np.stack((x_val, y_val_error))

import matplotlib.pyplot as plt

plt.scatter(x_val, y_val_error)
plt.show()

np.save(open("challenge_data.npy", "wb"), arr)
