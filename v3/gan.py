import shutil
import glob
import tensorflow as tf
import network3
import phase_parameters.params
import tf_functions
import numpy as np
import matplotlib.pyplot as plt








if __name__ == "__main__":

    # init data object
    get_data = network3.GetData(batch_size=10)


    # run name
    # can do multiple run names for the same model
    run_name = 'gan_test_2111'


    # copy the model to a new version to use for unsupervised learning
    modelname = 'run1'
    for file in glob.glob(r'./models/{}.ckpt.*'.format(modelname)):
        file_newname = file.replace(modelname, modelname+'_gan')
        shutil.copy(file, file_newname)

    writer = tf.summary.FileWriter("./tensorboard_graph_gan/" + modelname+"_gan_"+run_name)


    tf_generator_graphs, streak_params = network3.initialize_xuv_ir_trace_graphs()
    nn_nodes = network3.setup_neural_net(streak_params)

    gan_mse_tb = tf.summary.scalar("gan_mse", nn_nodes["gan"]["gan_network_loss"])

    label_mses = []
    plt.ion()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./models/{}.ckpt".format(modelname + "_gan"))

        # train gan network
        for i in range(20000):

            #print("training")

            gan_input = np.random.rand(100).reshape(1, -1)
            sess.run(nn_nodes["gan"]["train"], feed_dict={nn_nodes["gan"]["learningrate"]: 0.0001,
                                                          nn_nodes["gan"]["gan_input"]: gan_input
                                                          })

            if i % 10 == 0:
                print(i)
                summ = sess.run(gan_mse_tb, feed_dict={nn_nodes["gan"]["gan_input"]: gan_input})
                writer.add_summary(summ, global_step=i + 1)
                writer.flush()



                # check mse between labels
                y_pred_out = sess.run(nn_nodes["y_pred"],
                                      feed_dict={nn_nodes["gan"]["gan_input"]: gan_input})
                y_pred_out = y_pred_out.reshape(-1)
                gan_label_out = sess.run(nn_nodes["gan"]["gan_label"],
                                         feed_dict={nn_nodes["gan"]["gan_input"]: gan_input})
                gan_label_out = gan_label_out.reshape(-1)
                label_mse_error = (1/len(gan_label_out)) * np.sum((y_pred_out - gan_label_out)**2)
                print("label mse: ", label_mse_error)
                label_mses.append(label_mse_error)
                plt.figure(1)
                plt.cla()
                plt.plot(label_mses)
                # plot the averages
                #avg = (1/(len(label_mses)))*np.sum(np.array(label_mses))
                #plt.plot([0, len(label_mses)], [avg, avg], color="red")
                plt.pause(0.001)


        #traces, labels = get_data.evaluate_on_test_data()
        #out = sess.run(nn_nodes["reconstruction"]["trace"], feed_dict={nn_nodes["trace_in"]: traces[0].reshape(1, -1)})

        gan_input = np.random.rand(100).reshape(1, -1)

        out = sess.run(nn_nodes["trace_in_image"], feed_dict={nn_nodes["gan"]["gan_input"]: gan_input})
        plt.figure(1)
        plt.pcolormesh(out)

        # see the labels
        print("------labels------")
        print(sess.run(nn_nodes["y_pred"], feed_dict={nn_nodes["gan"]["gan_input"]: gan_input}))
        print(sess.run(nn_nodes["gan"]["gan_label"], feed_dict={nn_nodes["gan"]["gan_input"]: gan_input}))
        print("------labels------")


        out = sess.run(nn_nodes["gan"]["xuv_E_prop"]["f_cropped"], feed_dict={nn_nodes["gan"]["gan_input"]: gan_input})
        plt.figure(2)
        plt.plot(np.real(out[0]), color="blue")
        plt.plot(np.imag(out[0]), color="red")

        out = sess.run(nn_nodes["reconstruction"]["trace"], feed_dict={nn_nodes["gan"]["gan_input"]: gan_input})
        plt.figure(3)
        plt.pcolormesh(out)

        out = sess.run(nn_nodes["reconstruction"]["xuv_E_pred_prop"]["f_cropped"], feed_dict={nn_nodes["gan"]["gan_input"]: gan_input})
        plt.figure(4)
        plt.plot(np.real(out[0]), color="blue")
        plt.plot(np.imag(out[0]), color="red")

        plt.ioff()
        plt.show()


        exit(0)
