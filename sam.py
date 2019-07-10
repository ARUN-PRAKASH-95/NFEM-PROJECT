import numpy as np
import matplotlib.pyplot as plt

a= np.array([1,2,3,4,5])
b= np.array([2,3,4,5,6])

fig, ax = plt.subplots()
for i in range(3):
    b = b+1
  
    ax.plot(a,b)
plt.show()