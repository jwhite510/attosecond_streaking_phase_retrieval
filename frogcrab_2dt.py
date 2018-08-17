import numpy as np
import matplotlib.pyplot as plt







def test_dipole_moment(p, ir, I_p):

    print('max p : ', np.max(p))
    p = np.linspace(0, np.max(p), 100)
    p_m_2d, tau_m_2d= np.meshgrid(p, ir.t)
    E_ir_m = np.ones_like(tau_m_2d) * ir.E_t.reshape(-1, 1)
    A_tau = -1*ir.dt*np.cumsum(E_ir_m, 0)
    d_m_numerator = p_m_2d + A_tau
    d_m_denom = ((p_m_2d + A_tau)**2 + 2*I_p)**3
    IR_d_vector = d_m_numerator / d_m_denom

    _, ax_d = plt.subplots(1, 1)
    ax_d.pcolormesh(ir.t, p, np.transpose(IR_d_vector), cmap='jet')
    ax_d.set_xlabel('tau')
    ax_d.set_ylabel('p')
    ax_d.set_title('test IR_d_vector')









def plot_t_cut(indexes, axis, products, number, product, axis_p, tau, p):
    #index[p, tau, time]

    axis[0][number].plot(np.real(product[indexes[0], indexes[1], :]), color='blue')
    axis[0][number].plot(np.imag(product[indexes[0], indexes[1], :]), color='red')

    # integrate and list number
    integral = np.abs(xuv.dt * np.sum(product[indexes[0], indexes[1], :]))**2
    axis[0][number].text(0.1, 1.1, 'integral: {}'.format(integral), transform=axis[0][number].transAxes)

    axis[1][number].plot(products['E_xuv_m'][indexes[0], indexes[1], :], color='blue', alpha=0.5, label='E_xuv_m')
    axis[1][number].plot(products['IR_d_3d'][indexes[0], indexes[1], :], color='teal', alpha=0.5, label='IR_d_3d')
    axis[1][number].plot(np.real(products['phi_p_t_3d'][indexes[0], indexes[1], :]), color='orange', alpha=0.5, label='phi_p_t_3d')
    axis[1][number].plot(np.imag(products['phi_p_t_3d'][indexes[0], indexes[1], :]), color='orange', linestyle='dashed', alpha=0.5)
    axis[1][number].plot(np.real(products['e_ft'][indexes[0], indexes[1], :]), color='red', alpha=0.5, label='e_ft')
    axis[1][number].plot(np.imag(products['e_ft'][indexes[0], indexes[1], :]), color='red', linestyle='dashed', alpha=0.5)
    if number == 3:
        axis[1][number].legend(bbox_to_anchor=(1, 0.8))

    # label the plots with corresponding numbers in the streaking trace and time axes traces
    for vertplot in [0, 1]:
        text = axis[vertplot][number].text(0.1, 0.9, number, transform=axis[vertplot][number].transAxes)
        text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))

    text = axis_p.text(tau[indexes[1]], p[indexes[0]], number, backgroundcolor='white', alpha=0.5)
    text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))
    S[indexes[0], indexes[1]] = 0

def attosecond_streak(xuv, ir, I_p, plot):

    # print('xuv.F:\n', xuv.F, '\n')
    mid_index = int(len(xuv.F)/2)

    # p = (2*(2*np.pi * xuv.F[mid_index+1:] - I_p))**0.5
    p = (4 * np.pi * xuv.F[mid_index+1:])**0.5


    # print(p)
    # print(p[3]-p[2])
    # print(p[5]-p[4])
    tau = ir.t
    global tau_m
    global p_m
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
    print('A_tau avg:', np.average(A_tau))
    # print('A_tau:\n', A_tau, '\n')
    A_tau_reverse_int = ir.dt * np.flip(np.cumsum(np.flip(A_tau, 0), 0), 0)
    # print('A_t_reverse_int:\n', A_tau_reverse_int, '\n')
    A_tau_reverse_int_square = ir.dt * np.flip(np.cumsum(np.flip(A_tau**2, 0), 0), 0)
    # print('A_tau_reverse_int_square:\n', A_tau_reverse_int_square, '\n')

    # construct 3d d vector
    d_m_numerator = p_m_2d + A_tau
    d_m_denom = ((p_m_2d + A_tau)**2 + 2*I_p)**3
    print('avg denom: ', np.average(d_m_denom))
    print('avg numerator', np.average(d_m_numerator))

    IR_d_vector = d_m_numerator / d_m_denom
    # print(IR_d_vector)
    # print(IR_d_vector)
    # print('IR_d_vector:\n', IR_d_vector, '\n')
    IR_d_3d = np.array([IR_d_vector]).swapaxes(0, 2) * np.ones_like(t_m)
    # print('IR_delay_3d:\n', IR_d_3d, '\n')


    test_dipole_moment(p, ir, I_p)
    # plt.show()
    # exit(0)

    # construct 3d exp(-i phi(p, t))
    phi_p_t = p_m_2d * A_tau_reverse_int + 0.5 * A_tau_reverse_int_square
    phi_p_t = np.exp(-1j * phi_p_t)
    phi_p_t_3d = np.array([phi_p_t]).swapaxes(0, 2) * np.ones_like(t_m)
    # print('phi_p_t_3d:\n', phi_p_t_3d, '\n')

    # construct exp(i (p^2/2 + I_p)t)
    e_ft = np.exp(1j * ((p_m**2)/2) * t_m)

    product = E_xuv_m * IR_d_3d  * phi_p_t_3d * e_ft

    # product =  IR_d_3d  * phi_p_t_3d

    integrated = xuv.dt * np.sum(product, 2)
    global S
    S = np.abs(integrated)**2




    # # view small section of IR d vector
    # _, ax = plt.subplots(1, 1)
    # ax.pcolormesh(np.transpose(IR_d_vector)[1:-1, 1:-1])
    # ax.set_title('small view')
    # # IR_d_vector[0:10, 0:10] = 10e100



    # plot a cross sectional view of dipole moment
    if plot==True:
        _, ax_d = plt.subplots(1, 4, figsize=(18, 5))
        ax_d[0].pcolormesh(ir.t, p, np.transpose(IR_d_vector), cmap='jet')
        ax_d[0].set_xlabel('tau')
        ax_d[0].set_ylabel('p')
        ax_d[0].set_title('IR_d_vector')

        # plot a cross sectional view of dipole moment

        ax_d[1].pcolormesh(ir.t, p, np.transpose(d_m_numerator), cmap='jet')
        ax_d[1].set_xlabel('tau')
        ax_d[1].set_ylabel('p')
        ax_d[1].set_title('d_m_numerator')

        # plot a cross sectional view of dipole moment

        ax_d[2].pcolormesh(ir.t, p, np.transpose(d_m_denom), cmap='jet')
        ax_d[2].set_xlabel('tau')
        ax_d[2].set_ylabel('p')
        ax_d[2].set_title('d_m_denom')

        # plot a cross sectional view of A tau
        ax_d[3].pcolormesh(ir.t, p, np.transpose(A_tau), cmap='jet')
        ax_d[3].set_xlabel('tau')
        ax_d[3].set_ylabel('p')
        ax_d[3].set_title('A_tau')


        # plot a cross sectional view of quantum phase term
        _, ax_p_t = plt.subplots(1, 1)
        ax_p_t.pcolormesh(ir.t, p, np.real(np.transpose(phi_p_t)), cmap='jet')
        ax_p_t.set_xlabel('tau')
        ax_p_t.set_ylabel('p')
        ax_p_t.set_title('phi_p_t')

        # plot the fourier transform matrix
        _, ax_e_ft = plt.subplots(1, 2, figsize=(10, 5))
        ax_e_ft[0].pcolormesh(t_m[0, 0, :],p_m[:, 0, 0], np.real(e_ft[:, 0, :]))
        ax_e_ft[1].pcolormesh(t_m[0, 0, :],p_m[:, 0, 0], np.imag(e_ft[:, 0, :]))
        ax_e_ft[0].set_title('real e_ft')
        ax_e_ft[1].set_title('imag e_ft')
        for i in [0, 1]:
            ax_e_ft[i].set_xlabel('time')
            ax_e_ft[i].set_ylabel('p')


        # plot Exuv(t-tau)
        _, ax_E_XUV = plt.subplots(1)
        ax_E_XUV.pcolormesh(t_m[0, 0, :], tau_m[0, :, 0], E_xuv_m[0, :, :])
        ax_E_XUV.set_xlabel('time')
        ax_E_XUV.set_ylabel('tau')

        fig, ax = plt.subplots(2, 4, figsize=(11, 5))

        # print('E_xuv_m:\n', E_xuv_m, '\n')

        products = {'E_xuv_m': E_xuv_m, 'IR_d_3d': IR_d_3d, 'phi_p_t_3d': phi_p_t_3d,
                    'e_ft': e_ft}

        # PLOT ir_d
        # print(np.shape(IR_d_3d))
        ax[1][0].plot(IR_d_3d[1, :, 0], color='blue')
        text = ax[1][0].text(0.1, 0.9, 'IR_d_3d', transform=ax[1][0].transAxes, color='blue')
        text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))
        #indexes=(p, tau)
        plot_t_cut(indexes=(0, 3), axis=ax, products=products, number=1, product=product, axis_p=ax[0][0],
                   tau=tau, p=p)

        plot_t_cut(indexes=(1, 3), axis=ax, products=products, number=2, product=product, axis_p=ax[0][0],
                   tau=tau, p=p)

        plot_t_cut(indexes=(2, 3), axis=ax, products=products, number=3, product=product, axis_p=ax[0][0],
                   tau=tau, p=p)


        ax[0][0].set_ylabel('p')
        ax[0][0].set_xlabel('tau')
        ax[0][0].pcolormesh(ir.t, p, S, cmap='jet')

    return ir.t, p, S








class Field():

    def __init__(self, N, w0, FWHM, tmax):


        self.dt = (2*tmax) / N
        self.t = self.dt * np.arange(-N/2, N/2, 1)
        self.E_t = np.real(np.exp(1j * w0 * self.t) * np.exp(-2*np.log(4) * (self.t**2) / FWHM**2))

        self.df = 1 / (N * self.dt)
        self.F = self.df * np.arange(-N/2, N/2, 1)




xuv = Field(N=128, w0=4e17, FWHM=10e-18, tmax=50e-18)
ir = Field(N=128, w0=1.5e14, FWHM=70e-15, tmax=100e-15)
ir.E_t = 1e23 * ir.E_t

# ir.E_t = np.array([1, 2, 3, 4])
# ir.dt = 1

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0][0].plot(ir.t/1e-15, ir.E_t/np.max(ir.E_t), color='blue')
ax[0][0].set_xlabel('time [Femtoseconds]')

ax[0][1].plot(xuv.t/1e-18, xuv.E_t, color='orange')
ax[0][1].set_xlabel('time [Attoseconds]')

ax[1][0].plot(ir.t/1e-15, ir.E_t/np.max(ir.E_t), color='blue')
ax[1][0].plot(xuv.t/1e-15, xuv.E_t, color='orange')
ax[1][0].set_xlabel('time [Femtoseconds]')

tau, p, S = attosecond_streak(xuv=xuv, ir=ir, I_p=0.8e18, plot=False)
ax[1][1].pcolormesh(tau/1e-15, p, S, cmap='jet')
ax[1][1].set_xlabel('delay [Femtoseconds]')
ax[1][1].set_ylabel('momentum')


plt.show()





