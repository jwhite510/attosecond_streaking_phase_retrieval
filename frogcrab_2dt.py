import numpy as np
import matplotlib.pyplot as plt




def plot_t_cut(indexes, axis, products):
    #index[p, tau, time]
    axis.plot(products['E_xuv_m'][indexes[0], indexes[1], :], color='blue', alpha=0.5)
    axis.plot(products['IR_d_3d'][indexes[0], indexes[1], :], color='teal', alpha=0.5)
    axis.plot(np.real(products['phi_p_t_3d'][indexes[0], indexes[1], :]), color='orange', alpha=0.5)
    axis.plot(np.imag(products['phi_p_t_3d'][indexes[0], indexes[1], :]), color='orange', linestyle='dashed', alpha=0.5)
    axis.plot(np.real(products['e_ft'][indexes[0], indexes[1], :]), color='red', alpha=0.1)
    axis.plot(np.imag(products['e_ft'][indexes[0], indexes[1], :]), color='red', linestyle='dashed', alpha=0.1)


def attosecond_streak(xuv, ir, I_p):

    # print('xuv.F:\n', xuv.F, '\n')
    mid_index = int(len(xuv.F)/2)
    p = (4 * np.pi * xuv.F[mid_index:])**2
    p = (p / 10e51)
    p = np.linspace(0, 1e6, 40)
    # print(p)
    # print(p[3]-p[2])
    # print(p[5]-p[4])

    tau_m, p_m, t_m = np.meshgrid(ir.t, p, xuv.t)

    E_xuv_m = np.ones_like(t_m) * xuv.E_t.reshape(1, 1, -1)
    # print('t_m:\n', t_m, '\n')
    # print('E_xuv_m:\n', E_xuv_m, '\n')
    # print('tau_m:\n', tau_m, '\n')
    # print('p_m:\n', p_m, '\n')

    # construct 2-d grid for delay and momentum from IR pulse
    p_m_2d, tau_m_2d= np.meshgrid(p, ir.t)
    E_ir_m = np.ones_like(tau_m_2d) * ir.E_t.reshape(-1, 1)
    # print('E_ir_m:\n', E_ir_m, '\n')
    # print('tau_m_2d:\n', tau_m_2d, '\n')
    # print('p_m_2d:\n', p_m_2d, '\n')

    # construct A(tau) integrals
    A_tau = -1*ir.dt*np.cumsum(E_ir_m, 0)
    # print('A_tau:\n', A_tau, '\n')
    A_tau_reverse_int = ir.dt * np.flip(np.cumsum(np.flip(A_tau, 0), 0), 0)
    # print('A_t_reverse_int:\n', A_tau_reverse_int, '\n')
    A_tau_reverse_int_square = ir.dt * np.flip(np.cumsum(np.flip(A_tau**2, 0), 0), 0)
    # print('A_tau_reverse_int_square:\n', A_tau_reverse_int_square, '\n')

    # construct 3d d vector
    d_m_numerator = p_m_2d + A_tau
    d_m_denom = ((p_m_2d + A_tau)**2 + 2*I_p)**3
    IR_d_vector = d_m_numerator / d_m_denom
    # print(IR_d_vector)

    # print('IR_d_vector:\n', IR_d_vector, '\n')
    IR_d_3d = np.array([IR_d_vector]).swapaxes(0, 2) * np.ones_like(t_m)
    # print('IR_delay_3d:\n', IR_d_3d, '\n')

    # construct 3d exp(-i phi(p, t))
    phi_p_t = p_m_2d * A_tau_reverse_int + 0.5 * A_tau_reverse_int_square
    phi_p_t = np.exp(-1j * phi_p_t)
    phi_p_t_3d = np.array([phi_p_t]).swapaxes(0, 2) * np.ones_like(t_m)
    # print('phi_p_t_3d:\n', phi_p_t_3d, '\n')

    # construct exp(i (p^2/2 + I_p)t)
    e_ft = np.exp(1j * ((p_m**2)/2 + I_p) * t_m)


    product = E_xuv_m * IR_d_3d * phi_p_t_3d * e_ft
    # product = phi_p_t_3d

    integrated = 1 * np.sum(product, 2)
    S = np.abs(integrated)**2
    fig, ax = plt.subplots(2, 2, figsize=(11, 5))
    ax[0][0].set_ylabel('p')
    ax[0][0].set_xlabel('tau')
    ax[0][0].pcolormesh(ir.t, p, S, cmap='jet')


    # print('E_xuv_m:\n', E_xuv_m, '\n')

    products = {'E_xuv_m': E_xuv_m, 'IR_d_3d': IR_d_3d, 'phi_p_t_3d': phi_p_t_3d,
                'e_ft': e_ft}

    # PLOT ir_d
    # print(np.shape(IR_d_3d))
    ax[1][0].plot(IR_d_3d[1, :, 0], color='pink')

    plot_t_cut(indexes=(32, 64), axis=ax[0][1], products=products)
    # plot_t_cut(indexes=(32, 60), axis=ax[1][1], products=products)

    plt.show()






class Field():

    def __init__(self, N, w0, FWHM, tmax):


        self.dt = (2*tmax) / N
        self.t = self.dt * np.arange(-N/2, N/2, 1)
        self.E_t = np.real(np.exp(1j * w0 * self.t) * np.exp(-2*np.log(4) * (self.t**2) / FWHM**2))

        self.df = 1 / (N * self.dt)
        self.F = self.df * np.arange(-N/2, N/2, 1)




xuv = Field(N=128, w0=1e18, FWHM=10e-18, tmax=30e-18)
ir = Field(N=128, w0=3e14, FWHM=70e-15, tmax=100e-15)
ir.E_t = ir.E_t * 10e18

# ir.E_t = np.array([1, 2, 3, 4])
# ir.dt = 1

fig, ax = plt.subplots(3, 1)
ax[0].plot(ir.t, ir.E_t, color='blue')
ax[1].plot(xuv.t, xuv.E_t, color='orange')

ax[2].plot(ir.t, ir.E_t, color='blue')
ax[2].plot(xuv.t, xuv.E_t*np.max(ir.E_t), color='orange')



attosecond_streak(xuv=xuv, ir=ir, I_p=20e10)













