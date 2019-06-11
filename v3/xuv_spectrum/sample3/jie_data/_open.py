import numpy as np
import pickle
import matplotlib.pyplot as plt
# with open('7002 Unstreaked vs retrieve electron spectrum.opj', 'rb') as file:
#     matrix = np.array(list(file))
#     matrix2 = np.array(list(file))
#     import ipdb; ipdb.set_trace() # BREAKPOINT
#     print("hello")
# exit()
# 
# with open('s3measured_spec.csv', 'rb') as file:
#     matrix = np.array(list(file))
#     import ipdb; ipdb.set_trace() # BREAKPOINT


with open("spec1_1.csv", "r") as file:
    mat11 = np.array(list(file))
    mat11_c0 = [float(e.strip("\n").split(",")[0])for e in mat11]
    mat11_c1 = [float(e.strip("\n").split(",")[1])for e in mat11]
    mat11_c2 = [float(e.strip("\n").split(",")[2])for e in mat11]
    mat11_c3 = [float(e.strip("\n").split(",")[3])for e in mat11]

with open("spec1_2.csv", "r") as file:
    mat12 = np.array(list(file))
    mat12_c0 = [float(e.strip("\n").split(",")[0])for e in mat12]
    mat12_c1 = [float(e.strip("\n").split(",")[1])for e in mat12]
    mat12_c2 = [float(e.strip("\n").split(",")[2])for e in mat12]
    mat12_c3 = [float(e.strip("\n").split(",")[3])for e in mat12]

with open("spec2_1.csv", "r") as file:
    mat21 = np.array(list(file))
    mat21_c0 = [float(e.strip("\n").split(",")[0])for e in mat21[1:]]
    mat21_c1 = [float(e.strip("\n").split(",")[1])for e in mat21[1:]]
    mat21_c2 = [float(e.strip("\n").split(",")[2])for e in mat21[1:]]

with open("spec2_2.csv", "r") as file:
    mat22 = np.array(list(file))
    mat22_c0 = [float(e.strip("\n").split(",")[0])for e in mat22[1:]]
    mat22_c1 = [float(e.strip("\n").split(",")[1])for e in mat22[1:]]
    mat22_c2 = [float(e.strip("\n").split(",")[2])for e in mat22[1:]]
    mat22_c3 = [float(e.strip("\n").split(",")[3])for e in mat22[1:]]
    mat22_c4 = [float(e.strip("\n").split(",")[4])for e in mat22[1:]]
    mat22_c5 = [float(e.strip("\n").split(",")[5])for e in mat22[1:]]
    mat22_c6 = [float(e.strip("\n").split(",")[6])for e in mat22[1:]]

plt.figure(1)
plt.plot(mat21_c1 ,mat21_c2)
plt.figure(2)
plt.plot(mat22_c1, mat22_c2)

# photon corrected spectrum
plt.figure(3)
plt.plot(mat11_c1, mat11_c2)
axtwin = plt.gca().twinx()
axtwin.plot(mat11_c1, mat11_c3)

#plot the measured spectrum and smoothed measured spectrum
plt.figure(4)
plt.plot(mat21_c1 ,mat21_c2, color="blue")
plt.plot(mat22_c1, mat22_c2, color="red")
plt.plot(mat11_c1, mat11_c2, color="green")
with open("spec.p", "wb") as file:
    obj = dict()
    obj["electron"] = dict()
    obj["electron"]["eV"] = mat21_c1
    # change this one to be the measured spectrum
    obj["electron"]["I"] = mat21_c2
    obj["photon"] = dict()
    obj["photon"]["eV"] = mat11_c1
    obj["photon"]["I"] = mat11_c2
    pickle.dump(obj, file)
    print("pickled")
plt.show()

