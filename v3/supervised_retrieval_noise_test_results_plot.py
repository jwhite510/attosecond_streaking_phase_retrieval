import numpy as np
import matplotlib.pyplot as plt
import pickle
import tf_functions




if __name__ == "__main__":

    with open("./supervised_retrieval_noise_test.p", "rb") as file:
        obj = pickle.load(file)

    print(obj.keys())
    print("type(obj['measured'])", type(obj['measured']))
    print("type(obj['retrieved'])", type(obj['retrieved']))

    for measured, retrieved, count_num in zip(obj["measured"], obj["retrieved"], obj["count_num"]):

        print("count_num", count_num)

        print("type(measured)", type(measured))
        print("type(retrieved)", type(retrieved))
        print("type(count_num)", type(count_num))
        exit()

