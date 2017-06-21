import numpy as np
import matplotlib.pyplot as plt

p = np.loadtxt('nparts')
plt.plot(p)

Ni_max = 3000000
plt.axhline(Ni_max,color='red')
plt.axhline(0.95*Ni_max, color='green', linestyle='--')
plt.axhline(0.9*Ni_max, color='green', linestyle='-.')

plt.title('Ion Tracker')
plt.xlabel('Time Step')
plt.ylabel('Number of Ions')
plt.ylim([0,1.1*Ni_max])

plt.show()
