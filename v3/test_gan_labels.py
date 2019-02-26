import shutil
import glob
import tensorflow as tf
import network3
import phase_parameters.params
import tf_functions
import numpy as np
import matplotlib.pyplot as plt




def plot_output(out, label):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.real(out[0]), color="blue")
    ax.plot(np.imag(out[0]), color="red")
    ax.plot(np.abs(out[0]), color="black", alpha=0.5)
    axtwin = ax.twinx()
    axtwin.plot(np.unwrap(np.angle(out[0])), color="green")

    # plot xuv coefs
    slide = 0
    for coeflabel in ["1st", "2nd", "3rd", "4th", "5th"]:
        axtwin.text(0.6+slide, 0.95, coeflabel, transform=axtwin.transAxes, backgroundcolor="white")
        slide += 0.05
    axtwin.text(0.6, 0.90, str(np.round(label[:, 0:5], 3)), transform=axtwin.transAxes, backgroundcolor="white")

    slide = 0
    for coeflabel in ["$\phi_0$", "$\lambda$", r"$\tau$", "$I_0$"]:
        axtwin.text(0.6+slide, 0.85, coeflabel, transform=axtwin.transAxes, backgroundcolor="white")
        slide += 0.05
    axtwin.text(0.6, 0.80, str(np.round(label[:, 5:], 3)), transform=axtwin.transAxes, backgroundcolor="white")







if __name__ == "__main__":











    # y_pred
    y_pred = np.array([[
        0.0024748, 0.18352431, -0.58629531, -0.08329642, -1.20897138, 0.08354165,
        1.92034864, -0.13649754, -1.02133417
    ]])

    # gan label
    gan_label = np.array([[
        0., -0.00180609, -0.05556788, 0.01049891, -0.18871063, 0.10408029,
        -0.99997032, -0.15304415, -0.99997729
    ]])

    mse = (1/len(gan_label[0])) * np.sum((y_pred - gan_label)**2)
    print("mse: ", mse)


    tf_generator_graphs, streak_params = network3.initialize_xuv_ir_trace_graphs()
    assert isinstance(streak_params, object)
    nn_nodes = network3.setup_neural_net(streak_params)

    with tf.Session() as sess:

        # verify output
        modelname = 'run1'
        saver = tf.train.Saver()
        saver.restore(sess, "./models/{}.ckpt".format(modelname))
        out = sess.run(nn_nodes["y_pred"], feed_dict={nn_nodes["gan"]["gan_xuv_out_nolin"]: gan_label[:, 0:5],
                                                      nn_nodes["gan"]["gan_ir_out"]: gan_label[:, 5:]})
        print(out)




        out = sess.run(nn_nodes["reconstruction"]["xuv_E_pred_prop"]["f_cropped"], feed_dict={nn_nodes["y_pred"]: y_pred})
        plot_output(out, y_pred)





        out = sess.run(nn_nodes["reconstruction"]["xuv_E_pred_prop"]["f_cropped"],feed_dict={nn_nodes["y_pred"]: gan_label})
        plot_output(out, gan_label)

        plt.show()


    
    






