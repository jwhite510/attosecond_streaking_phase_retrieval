import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

k = 2
# t = np.array([0, 1, 2, 3, 4, 5, 6, 7])
t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])# shape t = n+k+1
print('np.shape(t): ', np.shape(t))


# c = np.array([-1, 2, 0, 5])
c = np.array([1, 2, 1, 2, 1, 2, 1]) # n = 7
print('np.shape(c): ', np.shape(c))
spl = BSpline(t, c, k)


fig, ax = plt.subplots()
xx = np.linspace(1, 10, 50)

ax.plot(xx, spl(xx))
plt.show()






# define know values for 


















