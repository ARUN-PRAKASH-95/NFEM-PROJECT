import numpy as np
import matplotlib.pyplot as plt

Time_step =np.array([0.01,0.025,0.05,0.075,0.1])
No_of_elements =np.array([143,71,46,37,32])
fig,ax = plt.subplots()
ax.plot(No_of_elements,Time_step)
ax.set_title('LoadStep $V_S$ No of elements')
ax.set_xlabel('No_of_elements')
ax.set_ylabel('LoadStep')
plt.savefig('Trend.png')