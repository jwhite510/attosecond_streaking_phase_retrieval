import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

# define parameter space
t = np.array([-0.2, -0.1, 0, 0.1, 0.2])
tau = np.array([-2, -1, 0, 1, 2])
p = np.array([-20, -10, 0, 10, 20])
E_irt = np.array([1, 2, 3, 4, 5])
E_xuv = np.array([0.1, 0.25, 0.34, 0.44, 0.56])


dt = 1
dp = 10

tau_m, p_m, t_m = np.meshgrid(tau, p, t)
print('t_m:\n', t_m, '\n')
print('tau_m:\n', tau_m, '\n')
print('p_m:\n', p_m, '\n')

# define complex IR vec
# E_irt = np.array([0.1+0.15j, 0.2+0.2j, 0.4+0.2j])

E_irt_m = np.ones_like(t_m) * E_irt.reshape(1, 1, -1)
print('E_irt_m:\n', E_irt_m, '\n')

# define A(t)
A_t = -1*dt*np.cumsum(E_irt_m, 2)
print('A_t:\n ', A_t, '\n')

# define phi_p_t
A_t_reverse_int = dt * np.flip(np.cumsum(np.flip(A_t, 2), 2), 2)
A_t_reverse_int_square = dt * np.flip(np.cumsum(np.flip(A_t**2, 2), 2), 2)
print('A_t_reverse_int:\n', A_t_reverse_int, '\n')


phi_p_t = p_m * A_t_reverse_int + 0.5 * A_t_reverse_int_square
print('phi_p_t:\n', phi_p_t, '\n')


# define Exuv(t-tau)
print('E_xuv:\n', E_xuv, '\n')

E_padded = np.pad(E_xuv[::-1], (0, len(E_xuv) - 1), mode='constant')
# print('E_padded:\n', E_padded, '\n')

toeplitz_m = np.tril(toeplitz(E_padded, E_xuv))
# crop the toeplitz matrix to match tau matrix
# print('toeplitz shape:\n', np.shape(toeplitz_m), '\n')
# print('toeplitz:\n', toeplitz_m, '\n')
# print('tau_m:\n', tau_m, '\n')

length_toepl, _ = np.shape(toeplitz_m)
# print('length_toepl:\n', length_toepl, '\n')
# print('middle index:', int((length_toepl-1)/2), '\n')
middle_index = int((length_toepl-1)/2)
_, crop_span, _ = np.shape(tau_m)
crop_side = int((crop_span-1)/2)

# print('crop_span:', crop_span, '\n')
# print('crop_side:', crop_side, '\n')
toeplitz_cropped = toeplitz_m[middle_index-crop_side:middle_index+crop_side+1, :]
# print('toeplitz_cropped:\n', toeplitz_cropped, '\n')
toeplitz_cropped_flipped = np.flip(toeplitz_cropped, 0)
print('toeplitz_cropped_flipped:\n', toeplitz_cropped_flipped, '\n')

p_length, _, _ = np.shape(p_m)
E_xuv_delay = np.array([toeplitz_cropped_flipped, ] * p_length)
print('E_xuv_delay:\n', E_xuv_delay, '\n')
# print('p_length:\n', p_length, '\n')
print('E_xuv_delay shape:\n', np.shape(E_xuv_delay), '\n')





