import shutil
import glob
import tensorflow as tf
import network3
import phase_parameters.params
import tf_functions
import numpy as np
import matplotlib.pyplot as plt




def plot_output(sess, label, name):



    # get output field
    field = sess.run(nn_nodes["reconstruction"]["xuv_E_pred_prop"]["f_cropped"], feed_dict={nn_nodes["y_pred"]: label})
    trace = sess.run(nn_nodes["reconstruction"]["trace"], feed_dict={nn_nodes["y_pred"]: label})
    coefs_divided_by_integral = sess.run(nn_nodes["reconstruction"]["xuv_E_pred_prop"]["coefs_divided_by_int"], feed_dict={nn_nodes["y_pred"]: label})

    coefs_divided_by_integral = np.array(coefs_divided_by_integral[:, :], dtype=np.float64)
    #vec2 = np.array(label[:, 0:5], dtype=np.float64)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(name+" XUV Field")
    ax.plot(np.real(field[0]), color="blue")
    ax.plot(np.imag(field[0]), color="red")
    ax.plot(np.abs(field[0]), color="black", alpha=0.5)
    axtwin = ax.twinx()
    axtwin.plot(np.unwrap(np.angle(field[0])), color="green")

    # plot xuv coefs
    vertical = 0.95
    slide = 0
    for coeflabel in ["1st", "2nd", "3rd", "4th", "5th"]:
        axtwin.text(0.6+slide, vertical, coeflabel, transform=axtwin.transAxes, backgroundcolor="white")
        slide += 0.08
    axtwin.text(0.6, vertical-0.05, str(np.round(label[:, 0:5], 3)), transform=axtwin.transAxes, backgroundcolor="white")

    # plot IR values
    vertical = 0.85
    slide = 0
    for coeflabel in ["$\phi_0$", "$\lambda$", r"$\tau$", "$I_0$"]:
        axtwin.text(0.6+slide, vertical, coeflabel, transform=axtwin.transAxes, backgroundcolor="white")
        slide += 0.08
    axtwin.text(0.6, vertical-0.05, str(np.round(label[:, 5:], 3)), transform=axtwin.transAxes, backgroundcolor="white")


    # plot xuv coefs
    vertical = 0.65
    slide = 0
    for coeflabel in ["1st", "2nd", "3rd", "4th", "5th"]:
        axtwin.text(0.6 + slide, vertical, coeflabel, transform=axtwin.transAxes, backgroundcolor="white")
        slide += 0.08
    axtwin.text(0.6, vertical - 0.05, str(np.round(coefs_divided_by_integral[:, :], 3)), transform=axtwin.transAxes,
                backgroundcolor="white")


    ax = fig.add_subplot(gs[0, 1])
    ax.pcolormesh(trace, cmap="jet")

    ax.set_title(name+" Trace")








if __name__ == "__main__":











    # y_pred
    y_pred = np.array([[
        0.0024748, 0.18352431, -0.58629531, -0.08329642, -1.20897138, 0.08354165,
        1.92034864, -0.13649754, -1.02133417
    ]], dtype=np.float64)

    # gan label
    gan_label = np.array([[
        0., -0.00180609, -0.05556788, 0.01049891, -0.18871063, 0.10408029,
        -0.99997032, -0.15304415, -0.99997729
    ]], dtype=np.float64)


    mse = (1/len(gan_label[0])) * np.sum((y_pred - gan_label)**2)
    print("mse: ", mse)


    tf_generator_graphs, streak_params = network3.initialize_xuv_ir_trace_graphs()
    nn_nodes = network3.setup_neural_net(streak_params)

    with tf.Session() as sess:

        # verify output
        modelname = 'run1'
        saver = tf.train.Saver()
        saver.restore(sess, "./models/{}.ckpt".format(modelname))
        out = sess.run(nn_nodes["y_pred"], feed_dict={nn_nodes["gan"]["gan_xuv_out_nolin"]: gan_label[:, 0:5],
                                                      nn_nodes["gan"]["gan_ir_out"]: gan_label[:, 5:]})
        print(out)



        # plot input field and trace
        # plot y_pred and reconstruction

        plot_output(sess, label=y_pred, name="Predicted")

        #out = sess.run(nn_nodes["reconstruction"]["xuv_E_pred_prop"]["f_cropped"],feed_dict={nn_nodes["y_pred"]: gan_label})
        plot_output(sess, label=gan_label, name="Input")

        plt.show()


    
    






