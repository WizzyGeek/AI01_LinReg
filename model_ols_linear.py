import numpy as np

fd = open("challenge_data.npy", "rb")
arr = np.load(fd)

x = arr[0]
y = arr[1]

mx = np.mean(x)
my = np.mean(y)

dx = x - mx

b = np.dot(dx, y - my) / np.sum(np.square(dx)) # slope
c = my - mx * b # constant

print(b, c)