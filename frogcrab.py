import numpy as np
import matplotlib.pyplot as plt

# define parameter space
t = np.array([-0.1, 0, 0.1])
tau = np.array([ -1, 0, 1])
p = np.array([-10, 0, 10])
dt = 1
dp = 10

tau_m, p_m, t_m = np.meshgrid(tau, p, t)
print('t_m:\n', t_m, '\n')
print('tau_m:\n', tau_m, '\n')
print('p_m:\n', p_m, '\n')

# define complex IR vec
# E_irt = np.array([0.1+0.15j, 0.2+0.2j, 0.4+0.2j])
E_irt = np.array([1, 2, 3])


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
E_xuv = np.array([0.1, 0.25, 0.34])


