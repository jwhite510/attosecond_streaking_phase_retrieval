import tensorflow as tf
import xuv_spectrum.spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial

# print(xuv_spectrum.spectrum.params.keys())
# exit(0)

# plt.figure(1)
# plt.plot(xuv_spectrum.spectrum.params["Ef"])
# plt.show()



def xuv_taylor_to_E(coef_values_normalized, coefs, amplitude):
    # print(coef_values_normalized)

    # eventually, will have to convert everything to atomic units before inputting here!!
    Ef = tf.constant(xuv_spectrum.spectrum.params["Ef"], dtype=tf.complex64)
    Ef = tf.reshape(Ef, [1, -1])

    fmat_taylor = tf.constant(xuv_spectrum.spectrum.params["fmat"]-xuv_spectrum.spectrum.params["f0"], dtype=tf.float32)

    f0 = tf.constant(xuv_spectrum.spectrum.params["f0"], dtype=tf.float32)

    # create factorials
    factorials = tf.constant(factorial(np.array(range(coefs))+1), dtype=tf.float32)
    factorials = tf.reshape(factorials, [1, -1, 1])

    # create exponents
    exponents = tf.constant(np.array(range(coefs))+1, dtype=tf.float32)

    # reshape the taylor fmat
    fmat_taylor = tf.reshape(fmat_taylor, [1, 1, -1])

    # reshape the exponential matrix
    exp_mat = tf.reshape(exponents, [1, -1, 1])

    # raise the fmat to the exponential power
    exp_mat_fmat = tf.pow(fmat_taylor, exp_mat)

    # scale the coefficients
    amplitude_mat = tf.constant(amplitude, dtype=tf.float32)
    amplitude_mat = tf.reshape(amplitude_mat, [1, -1, 1])

    # amplitude scales with exponent
    amplitude_scaler = tf.pow(amplitude_mat, exp_mat)

    # reshape the coef values and scale them
    coef_values = tf.reshape(coef_values_normalized, [tf.shape(coef_values_normalized)[0], -1, 1]) * amplitude_scaler

    # divide by the factorials
    coef_div_fact = tf.divide(coef_values, factorials)

    # multiply by the fmat
    taylor_coefs_mat = coef_div_fact * exp_mat_fmat

    # this is the phase angle, summed along the taylor terms
    taylor_terms_summed = tf.reduce_sum(taylor_coefs_mat, axis=1)

    # apply the phase angle to Ef
    Ef_prop = Ef * tf.exp(tf.complex(imag=taylor_terms_summed, real=tf.zeros_like(taylor_terms_summed)))


    with tf.Session() as sess:
        # input_array = np.array([[0.01, 0.02, 0.03, 0.04, 0.05],
        #                         [0.12, 0.1, 0.2, 0.13, 0.16]])
        input_array = np.array([[0.00, 0.5, 0.00, 0.00, 0.00],
                                [0.00, -0.5, 0.00, 0.00, 0.00]])
        # print(np.shape(sess.run(taylor_terms_summed, feed_dict={coefs_in: input_array})))
        out = sess.run(Ef_prop, feed_dict={coefs_in: input_array})

        plt.figure(1)
        plt.plot(np.real(out[0]), color="blue")
        plt.plot(np.imag(out[0]), color='red')
        plt.plot(np.abs(out[0]), color='black')
        axtwin = plt.gca().twinx()
        axtwin.plot(np.unwrap(np.angle(out[0])), color='green')

        plt.figure(2)
        plt.plot(np.real(out[1]), color="blue")
        plt.plot(np.imag(out[1]), color='red')
        plt.plot(np.abs(out[1]), color='black')
        axtwin = plt.gca().twinx()
        axtwin.plot(np.unwrap(np.angle(out[1])), color='green')



        plt.show()
        print(out)
        exit(0)



    print(coef_values)
    exit(0)

    print(coef_values_normalized)







    # with tf.Session() as sess:
    #     div1 = tf.constant(np.array([1, 2, 5]), dtype=tf.float32)
    #     div2 = tf.constant(np.array([1, 2, 2]), dtype=tf.float32)
    #     print(sess.run(tf.divide(div1, div2)))




    exit(0)




    exit(0)
    print(factorials)
    print(exponents)



    exit(0)
    print(coef_values_normalized)





    with tf.Session() as sess:
        print(sess.run(coefficients, feed_dict={coefs_in: np.array([0.5, 0.2, -0.2, 0.1, 0.2]).reshape(1, -1)}))
        # print(sess.run(Ef, feed_dict={coefs_in: np.array([0.5, 0.2, -0.2, 0.1, 0.2]).reshape(1, -1)}))
        print(sess.run(f0, feed_dict={coefs_in: np.array([0.5, 0.2, -0.2, 0.1, 0.2]).reshape(1, -1)}))





coefs_in = tf.placeholder(tf.float32, shape=[None, 5])
xuv_taylor_to_E(coefs_in, coefs=5, amplitude=12.0)

