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
    run_name = 'gan_test_20'


    # copy the model to a new version to use for unsupervised learning
    modelname = 'run1'
    for file in glob.glob(r'./models/{}.ckpt.*'.format(modelname)):
        file_newname = file.replace(modelname, modelname+'_gan')
        shutil.copy(file, file_newname)

    writer = tf.summary.FileWriter("./tensorboard_graph_gan/" + modelname+"_gan_"+run_name)


    tf_generator_graphs, streak_params = network3.initialize_xuv_ir_trace_graphs()
    nn_nodes = network3.setup_neural_net(streak_params)

    gan_mse_tb = tf.summary.scalar("gan_mse", nn_nodes["gan"]["gan_network_loss"])


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


        #traces, labels = get_data.evaluate_on_test_data()
        #out = sess.run(nn_nodes["reconstruction"]["trace"], feed_dict={nn_nodes["trace_in"]: traces[0].reshape(1, -1)})

        gan_input = np.random.rand(100).reshape(1, -1)

        out = sess.run(nn_nodes["trace_in_image"], feed_dict={nn_nodes["gan"]["gan_input"]: gan_input})
        plt.figure(1)
        plt.pcolormesh(out)

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

        plt.show()


        exit(0)
