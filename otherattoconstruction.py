import numpy as np
import matplotlib.pyplot as plt




c = 3e8
eps0 = 8.85e-12

# % Input parameters (x == x-ray photon)
ionization_potential_eV = 24.6
central_photon_x_energy_eV = 800
photon_IR_energy_eV = 1.24/2.5
field_intensity_W_per_m2 = 2e12*100**2
field_V_per_m = np.sqrt(2*field_intensity_W_per_m2/(c*eps0))
tau_IR_as = 12e3
K_max_eV = 85



# % Uncomment for chirped pulse
# tau_x_as = 342.2
# b_per_as2 = 1.66e-4

# % Uncomment for unchirped pulse
tau_x_as = 24.6
b_per_as2 = 0

#  conversion functions
def energy_eV_to_au(energy_eV):
    return energy_eV / 27.21
def eneergy_au_to_eV(energy_au):
    return energy_au*27.21
def field_SI_to_au(field_SI):
    return field_SI/5.14e11
def field_au_to_SI(field_au):
    return field_au*5.14e11
def time_as_to_au(time_as):
    return time_as / 24.2
def time_au_to_as(time_au):
    return time_au/24.2
au_per_as = 1/24.2

tau_IR_au = tau_IR_as*au_per_as
tau_x_au = tau_x_as * au_per_as

a_au = 4 * np.log(2) / (tau_x_au)**2
b_au = b_per_as2/(au_per_as**2)

# % Calculate some parameters in atomic units
ionization_potential_au = energy_eV_to_au(ionization_potential_eV)
central_photon_x_energy_au = energy_eV_to_au(central_photon_x_energy_eV)
central_kinetic_energy_au = central_photon_x_energy_au - ionization_potential_au
angular_frequency_IR_au = energy_eV_to_au(photon_IR_energy_eV)
angular_frequency_x_au = central_photon_x_energy_au
peak_IR_field_au = field_SI_to_au(field_V_per_m)
K_max_au = energy_eV_to_au(K_max_eV)


# % Define functions for the kinetic energy and fields
def kinetic_energy_au(photon_x_energy_au):
    return photon_x_energy_au - ionization_potential_au

def IR_electric_field_au(t_au):
    return peak_IR_field_au*np.real(np.exp(-2*np.log(2)*t_au**2/tau_IR_au**2 + 1j*angular_frequency_IR_au*t_au))

def x_electric_field_au(a_au, b_au, t_au):
    return np.real(np.exp(-a_au*t_au**2 + 1j*b_au*t_au**2 + 1j*angular_frequency_x_au*t_au))

# % Define the simulation time and the field of the x-ray wave
t_au = 1.5*tau_x_au*np.linspace(-1,1, 2000)
x_field_au = x_electric_field_au(a_au, b_au, t_au)

# % Define the delay time and K, which will be the axes of the final
# % streaking trace
tau_d_au = 2*tau_IR_au*np.linspace(-1,1,120)
K_au = central_kinetic_energy_au + ionization_potential_au + 2.5*K_max_au*np.linspace(-1,1, 120)

# % Create a grid of points for faster calculation
# print(np.shape(t_au))
# print(np.shape(K_au))
# exit(0)

# [T_AU, K_AU] = np.meshgrid(t_au, K_au)
[K_AU, T_AU] = np.meshgrid(K_au, t_au)
V_AU = np.sqrt(2*K_AU)



plt.figure(1)
plt.plot(t_au, x_field_au, color='blue')
delay_index_to_plot = 50
field_IR_au = IR_electric_field_au(T_AU + tau_d_au[delay_index_to_plot])
plt.plot(t_au, np.real(np.exp(1j*V_AU[:,delay_index_to_plot].reshape(-1, 1)/angular_frequency_IR_au**2*field_IR_au)), color='red')



# % Compute the electron density
S = np.zeros((len(tau_d_au), len(K_au)))
field_x_au = x_electric_field_au(a_au, b_au, T_AU)

for ind in range(len(tau_d_au)):

    field_IR_au = IR_electric_field_au(T_AU + tau_d_au[ind])
    phi = V_AU / angular_frequency_IR_au ** 2. * field_IR_au
    S_1 = field_x_au * (V_AU - field_IR_au/angular_frequency_IR_au) * np.exp(1j*phi) * np.exp(-1j*K_AU*T_AU)
    S[ind, :] = np.abs(np.trapz(S_1, axis=0))**2


plt.figure(2)
plt.pcolormesh(np.transpose(S), cmap='jet')

plt.show()








