import numpy as np
import matplotlib.pyplot as plt

x = range(100)
b = 30
colors = ["k" if i < b else "r" for i in range(len(list(x)))]

plt.figure()
plt.scatter(x,x, c=colors)
plt.show()
