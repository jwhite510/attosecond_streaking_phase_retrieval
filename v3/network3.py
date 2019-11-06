import tensorflow as tf
import tf_functions
import numpy as np
import sys
import scipy.constants as sc
import tables
import shutil
import matplotlib.pyplot as plt
import os
import phase_parameters.params
import measured_trace.get_trace as get_measured_trace
# import fake_measured_trace.get_fake_meas_trace as get_measured_trace
import unsupervised_retrieval


class PhaseNetTrain:
    def __init__(self, modelname):

        # convert the measured trace to a proof trace
        # also this function clears tf default graph

        # self.measured_trace = convert_regular_trace_to_proof(get_measured_trace.trace)
        self.measured_trace = get_measured_trace.trace

        # build neural net graph
        self.nn_nodes = setup_neural_net()

        self.measured_axes = unsupervised_retrieval.create_plot_axes()
        # create a feed dictionary to test on the measured trace
        self.measured_feed_dict = {
                self.nn_nodes["general"]["x_in"]: self.measured_trace.reshape(1, -1)
                }

        # test_generate_data(nn_nodes)

        print("built neural net")

        # init data object
        self.get_data = GetData(batch_size=10)

        # initialize mse tracking objects
        self.tf_loggers = init_tf_loggers(self.nn_nodes)


        # saver and set epoch number to run
        self.saver = tf.train.Saver()
        self.epochs = 80

        # set the name of the neural net test run and save the settigns
        self.modelname = modelname

        print('starting ' + self.modelname)

        shutil.copyfile('./network3.py', './models/network3_{}.py'.format(self.modelname))

        # create figures for showing results
        self.axes = {}

        self.axes["testplot1"], self.axes["testfig1"]= create_sample_plot()
        self.axes["testplot2"], self.axes["testfig2"]= create_sample_plot()

        self.axes["trainplot1"], self.axes["trainfig1"]= create_sample_plot()
        self.axes["trainplot2"], self.axes["trainfig2"]= create_sample_plot()

        # plt.ion()

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.writer = tf.summary.FileWriter("./tensorboard_graph/" + self.modelname)
        self.i = None
        self.epoch = None
        self.dots = None

    def supervised_learn(self):

        for self.i in range(self.epochs):
            self.epoch = self.i + 1
            print("Epoch : {}".format(self.epoch))

            # iterate through every sample in the training set
            self.dots = 0
            alternate_training_counter = 0
            while self.get_data.batch_index < self.get_data.samples:

                self.show_loading_bar()

                # retrieve data
                batch_x, batch_y = self.get_data.next_batch()

                # train only with coefficients
                self.sess.run(self.nn_nodes["supervised"]["phase_network_train_coefs_params"],
                         feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x,
                                    self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y,
                                    self.nn_nodes["general"]["hold_prob"]: 1.0,
                                    self.nn_nodes["supervised"]["s_LR"]: 0.0001})

                # train with coefficients then with fields
                # if self.i < 15:
                #     # train with only coefficients first
                #     self.sess.run(self.nn_nodes["supervised"]["phase_network_train_coefs_params"],
                #              feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x,
                #                         self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y,
                #                         self.nn_nodes["general"]["hold_prob"]: 0.8,
                #                         self.nn_nodes["supervised"]["s_LR"]: 0.0001})

                # else:
                #     # train with fields
                #     self.sess.run(self.nn_nodes["supervised"]["phase_network_train_fields"],
                #              feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x,
                #                         self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y,
                #                         self.nn_nodes["general"]["hold_prob"]: 0.8,
                #                         self.nn_nodes["supervised"]["s_LR"]: 0.0001})

            print("")
            self.add_tensorboard_values()
            # every x steps plot predictions
            if self.epoch % 20 == 0 or self.epoch <= 15:
                # update the plot

                self.update_plots()

                # save model
                self.saver.save(self.sess, "models/" + self.modelname + ".ckpt")

            if self.epoch % 5 == 0 or self.epoch==1:
                self.retrieve_experimental()


            # return the index to 0
            self.get_data.batch_index = 0

        self.saver.save(self.sess, "models/"+self.modelname+".ckpt")

    def add_tensorboard_values(self):

        #***********************************
        # ..................................
        # ..........test set...............
        # ..................................
        #***********************************
        # view the mean squared error of the test data
        batch_x_test, batch_y_test = self.get_data.evaluate_on_test_data()


        #---------------------------------
        # -------phase curve loss---------
        #---------------------------------
        print("Phasecurve test MSE: ", self.sess.run(self.nn_nodes["supervised"]["phase_network_phasecurve_loss"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test}))
        summ = self.sess.run(self.tf_loggers["test_mse_tb_phasecurve"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
        self.writer.add_summary(summ, global_step=self.epoch)

        # ---------------------------------
        # ----------fields loss------------
        # ---------------------------------
        print("fields test MSE: ", self.sess.run(self.nn_nodes["supervised"]["phase_network_fields_loss"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test}))
        summ = self.sess.run(self.tf_loggers["test_mse_tb_fields"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
        self.writer.add_summary(summ, global_step=self.epoch)

        # ---------------------------------
        # --------coef params loss---------
        # ---------------------------------
        print("coefs params test MSE: ", self.sess.run(self.nn_nodes["supervised"]["phase_network_coefs_params_loss"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test}))
        summ = self.sess.run(self.tf_loggers["test_mse_tb_coefs_params"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
        self.writer.add_summary(summ, global_step=self.epoch)

        # ----------------------------------------
        # -----------xuv avg loss-----------------
        # ----------------------------------------
        summ = self.sess.run(self.tf_loggers["individual"]["xuv_coefs_avg_test"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
        self.writer.add_summary(summ, global_step=self.epoch)
        # ----------------------------------------
        # -----------ir avg loss------------------
        # ----------------------------------------
        summ = self.sess.run(self.tf_loggers["individual"]["ir_avg_test"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
        self.writer.add_summary(summ, global_step=self.epoch)
        # ----------------------------------------
        # -----------xuv coefficient losses-------
        # ----------------------------------------
        for tens in self.tf_loggers["individual"]["xuv_coefs_test"]:
            summ = self.sess.run(tens, feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                        self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
            self.writer.add_summary(summ, global_step=self.epoch)
        # ---------------------------------------
        # ---------IR parameter loss-------------
        # ---------------------------------------
        for key in self.tf_loggers["individual"]["ir_params_test"]:
            tens = self.tf_loggers["individual"]["ir_params_test"][key]
            summ = self.sess.run(tens, feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_test,
                                        self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
            self.writer.add_summary(summ, global_step=self.epoch)


        #***********************************
        # ..................................
        # ..........train set................
        # ..................................
        #***********************************
        # view the mean squared error of the train data
        batch_x_train, batch_y_train = self.get_data.evaluate_on_train_data(samples=500)

        # ----------------------------------------
        # ----------phase curve loss--------------
        # ----------------------------------------
        print("Phasecurve train MSE: ", self.sess.run(self.nn_nodes["supervised"]["phase_network_phasecurve_loss"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train}))
        summ = self.sess.run(self.tf_loggers["train_mse_tb_phasecurve"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
        self.writer.add_summary(summ, global_step=self.epoch)


        #----------------------------------------
        # ----------fields loss------------------
        #----------------------------------------
        print("fields train MSE: ", self.sess.run(self.nn_nodes["supervised"]["phase_network_fields_loss"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train}))
        summ = self.sess.run(self.tf_loggers["train_mse_tb_fields"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
        self.writer.add_summary(summ, global_step=self.epoch)

        #-----------------------------------------
        # -----------coef params loss--------------
        #-----------------------------------------
        print("coefs params train MSE: ", self.sess.run(self.nn_nodes["supervised"]["phase_network_coefs_params_loss"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train}))
        summ = self.sess.run(self.tf_loggers["train_mse_tb_coefs_params"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
        self.writer.add_summary(summ, global_step=self.epoch)


        # ----------------------------------------
        # -----------xuv avg loss-----------------
        # ----------------------------------------
        summ = self.sess.run(self.tf_loggers["individual"]["xuv_coefs_avg_train"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
        self.writer.add_summary(summ, global_step=self.epoch)
        # ----------------------------------------
        # -----------ir avg loss------------------
        # ----------------------------------------
        summ = self.sess.run(self.tf_loggers["individual"]["ir_avg_train"],
                                            feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                            self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
        self.writer.add_summary(summ, global_step=self.epoch)
        # ----------------------------------------
        # -----------xuv coefficient losses-------
        # ----------------------------------------
        for tens in self.tf_loggers["individual"]["xuv_coefs_train"]:
            summ = self.sess.run(tens, feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                        self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
            self.writer.add_summary(summ, global_step=self.epoch)

        # ---------------------------------------
        # ---------IR parameter loss-------------
        # ---------------------------------------
        for key in self.tf_loggers["individual"]["ir_params_train"]:
            tens = self.tf_loggers["individual"]["ir_params_train"][key]
            summ = self.sess.run(tens, feed_dict={self.nn_nodes["supervised"]["x_in"]: batch_x_train,
                                        self.nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
            self.writer.add_summary(summ, global_step=self.epoch)

        # ..................................
        # .....write to tensorboard.........
        # ..................................
        self.writer.flush()

    def show_loading_bar(self):
        # display loading bar
        percent = 50 * self.get_data.batch_index / self.get_data.samples
        if percent - self.dots > 1:
            print(".", end="", flush=True)
            self.dots += 1

    def update_plots(self):

        # def update_plots(data_obj, sess, nn_nodes, modelname, epoch, axes):
        batch_x_train, batch_y_train = self.get_data.evaluate_on_train_data(samples=500)
        self.plot_predictions(x_in=batch_x_train, y_in=batch_y_train, indexes=[0, 10, 20], set='train_data_1',
                              axes=self.axes["trainplot1"], figure=self.axes["trainfig1"])
        self.plot_predictions(x_in=batch_x_train, y_in=batch_y_train, indexes=[30, 40, 50], set='train_data_2',
                              axes=self.axes["trainplot2"], figure=self.axes["trainfig2"])

        batch_x_test, batch_y_test = self.get_data.evaluate_on_test_data()
        self.plot_predictions(x_in=batch_x_test, y_in=batch_y_test, indexes=[0, 10, 20], set='test_data_1',
                              axes=self.axes["testplot1"], figure=self.axes["testfig1"])
        self.plot_predictions(x_in=batch_x_test, y_in=batch_y_test, indexes=[30, 40, 50], set='test_data_2',
                              axes=self.axes["testplot2"], figure=self.axes["testfig2"])

        plt.show()
        plt.pause(0.001)

    def plot_predictions(self, x_in, y_in, indexes, set, axes, figure):
        # def plot_predictions(x_in, y_in, indexes, axes, figure, epoch, set, net_name, nn_nodes, sess):
        # get find where in the vector is the ir and xuv
        print("plot predicitons")
        K_values = phase_parameters.params.K
        tau_values = phase_parameters.params.delay_values

        for j, index in enumerate(indexes):

            mse = self.sess.run(self.nn_nodes["supervised"]["phase_network_fields_loss"],
                                    feed_dict={self.nn_nodes["general"]["x_in"]: x_in[index].reshape(1, -1),
                                    self.nn_nodes["supervised"]["actual_coefs_params"]: y_in[index].reshape(1, -1)})

            # get the actual fields
            actual_xuv_field = self.sess.run(self.nn_nodes["supervised"]["supervised_label_fields"]["xuv_E_prop"]["f_cropped"],
                                    feed_dict={self.nn_nodes["supervised"]["actual_coefs_params"]: y_in[index].reshape(1, -1)})

            actual_ir_field = self.sess.run(self.nn_nodes["supervised"]["supervised_label_fields"]["ir_E_prop"]["f_cropped"],
                                    feed_dict={self.nn_nodes["supervised"]["actual_coefs_params"]: y_in[index].reshape(1, -1)})

            # get the predicted fields
            predicted_xuv_field = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["f_cropped"],
                                    feed_dict={self.nn_nodes["general"]["x_in"]: x_in[index].reshape(1, -1)})

            predicted_ir_field = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["ir_E_prop"]["f_cropped"],
                                    feed_dict={self.nn_nodes["general"]["x_in"]: x_in[index].reshape(1, -1)})

            actual_xuv_field = actual_xuv_field.reshape(-1)
            actual_ir_field = actual_ir_field.reshape(-1)
            predicted_xuv_field = predicted_xuv_field.reshape(-1)
            predicted_ir_field = predicted_ir_field.reshape(-1)

            # calculate generated streaking trace
            generated_trace = self.sess.run(self.nn_nodes["general"]["reconstructed_trace"],
                                    feed_dict={self.nn_nodes["general"]["x_in"]: x_in[index].reshape(1, -1)})

            axes[j]['input_trace'].cla()
            axes[j]['input_trace'].pcolormesh(x_in[index].reshape(len(K_values), len(tau_values)), cmap='jet')
            axes[j]['input_trace'].text(0.0, 1.0, 'input_trace', transform=axes[j]['input_trace'].transAxes,backgroundcolor='white')
            axes[j]['input_trace'].set_xticks([])
            axes[j]['input_trace'].set_yticks([])

            axes[j]['actual_xuv'].cla()
            axes[j]['actual_xuv_twinx'].cla()
            axes[j]['actual_xuv'].plot(np.real(actual_xuv_field), color='blue', alpha=0.3)
            axes[j]['actual_xuv'].plot(np.imag(actual_xuv_field), color='red', alpha=0.3)
            axes[j]['actual_xuv'].plot(np.abs(actual_xuv_field), color='black')
            # plot the phase
            axes[j]['actual_xuv_twinx'].plot(np.unwrap(np.angle(actual_xuv_field)), color='green')
            axes[j]['actual_xuv_twinx'].tick_params(axis='y', colors='green')
            axes[j]['actual_xuv'].text(0.0,1.0, 'actual_xuv', transform=axes[j]['actual_xuv'].transAxes, backgroundcolor='white')
            axes[j]['actual_xuv'].set_xticks([])
            axes[j]['actual_xuv'].set_yticks([])

            axes[j]['predict_xuv'].cla()
            axes[j]['predict_xuv_twinx'].cla()
            axes[j]['predict_xuv'].plot(np.real(predicted_xuv_field), color='blue', alpha=0.3)
            axes[j]['predict_xuv'].plot(np.imag(predicted_xuv_field), color='red', alpha=0.3)
            axes[j]['predict_xuv'].plot(np.abs(predicted_xuv_field), color='black')
            #plot the phase
            axes[j]['predict_xuv_twinx'].plot(np.unwrap(np.angle(predicted_xuv_field)), color='green')
            axes[j]['predict_xuv_twinx'].tick_params(axis='y', colors='green')
            axes[j]['predict_xuv'].text(0.0, 1.0, 'predict_xuv', transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
            axes[j]['predict_xuv'].text(-0.4, 0, 'MSE: {} '.format(str(mse)),
                                        transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
            axes[j]['predict_xuv'].set_xticks([])
            axes[j]['predict_xuv'].set_yticks([])

            axes[j]['actual_ir'].cla()
            axes[j]['actual_ir'].plot(np.real(actual_ir_field), color='blue')
            axes[j]['actual_ir'].plot(np.imag(actual_ir_field), color='red')
            axes[j]['actual_ir'].text(0.0, 1.0, 'actual_ir', transform=axes[j]['actual_ir'].transAxes, backgroundcolor='white')

            if j == 0:

                axes[j]['actual_ir'].text(0.5, 1.25, self.modelname, transform=axes[j]['actual_ir'].transAxes,
                                          backgroundcolor='white')

                axes[j]['actual_ir'].text(0.5, 1.1, set, transform=axes[j]['actual_ir'].transAxes,
                                          backgroundcolor='white')

            axes[j]['actual_ir'].set_xticks([])
            axes[j]['actual_ir'].set_yticks([])

            axes[j]['predict_ir'].cla()
            axes[j]['predict_ir'].plot(np.real(predicted_ir_field), color='blue')
            axes[j]['predict_ir'].plot(np.imag(predicted_ir_field), color='red')
            axes[j]['predict_ir'].text(0.0, 1.0, 'predict_ir', transform=axes[j]['predict_ir'].transAxes,backgroundcolor='white')
            axes[j]['predict_ir'].set_xticks([])
            axes[j]['predict_ir'].set_yticks([])

            axes[j]['reconstruct'].pcolormesh(generated_trace,cmap='jet')
            axes[j]['reconstruct'].text(0.0, 1.0, 'reconstructed_trace', transform=axes[j]['reconstruct'].transAxes,backgroundcolor='white')
            axes[j]['reconstruct'].set_xticks([])
            axes[j]['reconstruct'].set_yticks([])

            # save image
            dir = "./nnpictures/" + self.modelname + "/" + set + "/"
            if not os.path.isdir(dir):
                os.makedirs(dir)
            figure.savefig(dir + str(self.epoch) + ".png")

    def retrieve_experimental(self):


        ir_f = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["ir_E_prop"]["f_cropped"],feed_dict=self.measured_feed_dict)[0]
        xuv_f = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["f_cropped"],feed_dict=self.measured_feed_dict)[0]
        xuv_f_phase = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["phasecurve_cropped"],feed_dict=self.measured_feed_dict)[0]
        xuv_f_full = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["f"],feed_dict=self.measured_feed_dict)[0]
        xuv_t = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["t"],feed_dict=self.measured_feed_dict)[0]

        #================================================
        #==================INPUT TRACES==================
        #================================================

        # calculate INPUT Autocorrelation from input image
        input_auto = self.sess.run(self.nn_nodes["unsupervised"]["autocorrelate"]["input_image_autocorrelate"], feed_dict=self.measured_feed_dict)

        # calculate INPUT PROOF trace from input image
        input_proof = self.sess.run(self.nn_nodes["unsupervised"]["proof"]["input_image_proof"]["proof"], feed_dict=self.measured_feed_dict)

        #================================================
        #==================MEASURED TRACES===============
        #================================================

        # reconstructed regular trace from input image
        reconstructed = self.sess.run(self.nn_nodes["general"]["reconstructed_trace"],feed_dict=self.measured_feed_dict)

        # calculate reconstructed proof trace
        reconstructed_proof = self.sess.run(self.nn_nodes["unsupervised"]["proof"]["reconstructed_proof"]["proof"], feed_dict=self.measured_feed_dict)

        # calculate reconstructed autocorrelation trace
        reconstruced_auto = self.sess.run(self.nn_nodes["unsupervised"]["autocorrelate"]["reconstructed_autocorrelate"], feed_dict=self.measured_feed_dict)

        # measured/calculated from input traces
        input_traces = dict()
        input_traces["trace"] = self.measured_trace
        input_traces["proof"] = input_proof
        input_traces["autocorrelation"] = input_auto

        # reconstruction traces
        recons_traces = dict()
        recons_traces["trace"] = reconstructed
        recons_traces["proof"] = reconstructed_proof

        recons_traces["autocorrelation"] = reconstruced_auto
        unsupervised_retrieval.plot_images_fields(axes=self.measured_axes, traces_meas=input_traces, traces_reconstructed=recons_traces,
                           xuv_f=xuv_f, xuv_f_phase=xuv_f_phase, xuv_f_full=xuv_f_full, xuv_t=xuv_t, ir_f=ir_f, i=self.epoch,
                           run_name=self.modelname+"measured_retrieval_while_training", true_fields=False, cost_function="trace",
                           method="Training", save_data_objs=True)
        plt.pause(0.00001)

class GetData():
    def __init__(self, batch_size):

        self.batch_counter = 0
        self.batch_index = 0
        self.batch_size = batch_size
        self.train_filename = 'train3.hdf5'
        self.test_filename = 'test3.hdf5'

        hdf5_file = tables.open_file(self.train_filename, mode="r")
        self.samples = hdf5_file.root.noise_trace.shape[0]
        hdf5_file.close()

    def next_batch(self):

        # retrieve the next batch of data from the data source
        hdf5_file = tables.open_file(self.train_filename, mode="r")

        xuv_coefs = hdf5_file.root.xuv_coefs[self.batch_index:self.batch_index + self.batch_size, :]
        ir_params = hdf5_file.root.ir_params[self.batch_index:self.batch_index + self.batch_size, :]
        appended_label_batch = np.append(xuv_coefs, ir_params, 1)

        trace_batch = hdf5_file.root.noise_trace[self.batch_index:self.batch_index + self.batch_size, :]

        hdf5_file.close()

        self.batch_index += self.batch_size

        return  trace_batch, appended_label_batch


    def evaluate_on_test_data(self):

        # this is used to evaluate the mean squared error of the data after every epoch
        hdf5_file = tables.open_file(self.test_filename, mode="r")

        xuv_coefs = hdf5_file.root.xuv_coefs[:, :]
        ir_params = hdf5_file.root.ir_params[:, :]
        appended_label_batch = np.append(xuv_coefs, ir_params, 1)

        trace_batch = hdf5_file.root.noise_trace[:, :]

        hdf5_file.close()

        return trace_batch, appended_label_batch



    def evaluate_on_train_data(self, samples):

        # this is used to evaluate the mean squared error of the data after every epoch
        hdf5_file = tables.open_file(self.train_filename, mode="r")

        xuv_coefs = hdf5_file.root.xuv_coefs[:samples, :]
        ir_params = hdf5_file.root.ir_params[:samples, :]
        appended_label_batch = np.append(xuv_coefs, ir_params, 1)

        trace_batch = hdf5_file.root.noise_trace[:samples, :]

        hdf5_file.close()

        return trace_batch, appended_label_batch

def convert_ir_params(ir_params):
    """
    convert the ir parameters to include only
    the variables which are changing, the phase
    and the intensity
    """
    # convert network output to theta
    ir_values_scaled = tf_functions.ir_from_params(ir_params)["scaled_values"]
    # ir_values_scaled["phase"] # radians

    # index:
    # "phase", "clambda", "pulseduration", "I"
    # phase_tens = tf.reshape(ir_params[:,0], [-1, 1])

    # the angle that is not between 0 and 2pi
    # ir_values_scaled["phase"]
    cos_angle, sin_angle = divide_to_cos_sin(ir_values_scaled["phase"])

    cos_angle = tf.reshape(cos_angle, [-1, 1])
    sin_angle = tf.reshape(sin_angle, [-1, 1])
    intensity_tens = tf.reshape(ir_params[:,3], [-1, 1])
    pulse_duration_tens = tf.reshape(ir_params[:,2], [-1, 1])

    label = tf.concat([cos_angle, sin_angle], axis=1)
    label = tf.concat([label, intensity_tens], axis=1)
    label = tf.concat([label, pulse_duration_tens], axis=1)

    return label

def convert_regular_trace_to_proof(regular_trace):
    image_noisy_placeholder = tf.placeholder(tf.float32, shape=[301, 38])
    proof_trace = tf_functions.proof_trace(image_noisy_placeholder)["proof"]
    with tf.Session() as sess:
        proof_trace_gen = sess.run(proof_trace, feed_dict={image_noisy_placeholder:regular_trace})
    tf.reset_default_graph()
    return proof_trace_gen


def log_base(x, base, translate):
    return tf.log(x+translate) / tf.log(base)

def create_fields_label_from_coefs_params(actual_coefs_params):
    xuv_coefs_actual = actual_coefs_params[:, 0:phase_parameters.params.xuv_phase_coefs]
    ir_params_actual = actual_coefs_params[:, phase_parameters.params.xuv_phase_coefs:]
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_actual)
    ir_E_prop = tf_functions.ir_from_params(ir_params_actual)["E_prop"]
    xuv_ir_field_label = concat_fields(xuv=xuv_E_prop["f_cropped"], ir=ir_E_prop["f_cropped"])

    fields = {}
    fields["xuv_E_prop"] = xuv_E_prop
    fields["ir_E_prop"] = ir_E_prop
    fields["xuv_ir_field_label"] = xuv_ir_field_label
    fields["actual_coefs_params"] = actual_coefs_params
    fields["xuv_coefs_actual"] = xuv_coefs_actual
    fields["ir_params_actual"] = ir_params_actual

    return fields

def concat_fields(xuv, ir):
    xuv_concat = tf.concat([tf.real(xuv), tf.imag(xuv)], axis=1)
    ir_concat = tf.concat([tf.real(ir), tf.imag(ir)], axis=1)
    both_fields_concat = tf.concat([xuv_concat, ir_concat], axis=1)

    return both_fields_concat

def test_generate_data(nn_nodes):
    # generate a bunch of samples and test threshold value
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)
        ax3 = fig.add_subplot(gs[:, :])
        plt.ion()

        # define threshold
        xuv_coefs_in = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        xuv_t = sess.run(nn_nodes["gan"]["xuv_E_prop"]["t"],
                         feed_dict={nn_nodes["gan"]["gan_xuv_out_nolin"]: xuv_coefs_in})
        threshold = np.max(np.abs(xuv_t[0])) * phase_parameters.params.threshold_scaler

        for _ in range(999):
            gan_in = np.random.random(100).reshape(1, -1)

            out = sess.run(nn_nodes["gan"]["xuv_E_prop"]["t"],
                           feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})

            indexmax = phase_parameters.params.threshold_max_index
            indexmin = phase_parameters.params.threshold_min_index

            indexmin_value = np.max(np.abs(out[0, :indexmin]))
            indexmax_value = np.max(np.abs(out[0, indexmax:]))

            ax3.cla()
            ax3.plot(np.real(out[0]), color="blue")
            ax3.plot(np.abs(out[0]), color="black")
            ax3.plot([indexmin, indexmin], [indexmin_value, 0], color="red")
            ax3.plot([indexmax, indexmax], [indexmax_value, 0], color="red")
            ax3.plot([indexmin, indexmax], [threshold, threshold], color="orange", linestyle="dashed")

            # print(indexmin_value)
            # print(indexmax_value)
            # print(threshold)

            if indexmin_value > threshold or indexmax_value > threshold:
                print("exceeded threshold")
                plt.ioff()
                plt.show()
                exit(0)

            plt.pause(0.001)

def separate_xuv_ir_vec(xuv_ir_vec):
    xuv = xuv_ir_vec[0:5]
    ir = xuv_ir_vec[5:]

    return xuv, ir

def init_tf_loggers(nn_nodes):
    test_mse_tb_phasecurve = tf.summary.scalar("test_mse_phasecurve", nn_nodes["supervised"]["phase_network_phasecurve_loss"])
    train_mse_tb_phasecurve = tf.summary.scalar("train_mse_phasecurve", nn_nodes["supervised"]["phase_network_phasecurve_loss"])

    test_mse_tb_fields = tf.summary.scalar("test_mse_fields", nn_nodes["supervised"]["phase_network_fields_loss"])
    train_mse_tb_fields = tf.summary.scalar("train_mse_fields", nn_nodes["supervised"]["phase_network_fields_loss"])

    test_mse_tb_coefs_params = tf.summary.scalar("test_mse_coef_params", nn_nodes["supervised"]["phase_network_coefs_params_loss"])
    train_mse_tb_coefs_params = tf.summary.scalar("train_mse_coef_params", nn_nodes["supervised"]["phase_network_coefs_params_loss"])

    tf_loggers = {}
    tf_loggers["test_mse_tb_phasecurve"] = test_mse_tb_phasecurve
    tf_loggers["train_mse_tb_phasecurve"] = train_mse_tb_phasecurve
    tf_loggers["test_mse_tb_fields"] = test_mse_tb_fields
    tf_loggers["train_mse_tb_fields"] = train_mse_tb_fields
    tf_loggers["test_mse_tb_coefs_params"] = test_mse_tb_coefs_params
    tf_loggers["train_mse_tb_coefs_params"] = train_mse_tb_coefs_params

    # log indiivdually:
    # ir parameters (total avg error)
    # xuv parameters (total avg error)
    # xuv parameters (individual)
    
    
    tf_loggers["individual"] = {}
    tf_loggers["individual"]["xuv_coefs_avg_test"] = tf.summary.scalar("xuv_coefs_avg_test", nn_nodes["supervised"]["extra_losses"]["xuv_loss"])
    tf_loggers["individual"]["xuv_coefs_avg_train"] = tf.summary.scalar("xuv_coefs_avg_train", nn_nodes["supervised"]["extra_losses"]["xuv_loss"])

    tf_loggers["individual"]["ir_avg_test"] = tf.summary.scalar("ir_avg_test", nn_nodes["supervised"]["extra_losses"]["ir_loss"])
    tf_loggers["individual"]["ir_avg_train"] = tf.summary.scalar("ir_avg_train", nn_nodes["supervised"]["extra_losses"]["ir_loss"])

    # logger for coefficients in test data
    tf_loggers["individual"]["xuv_coefs_test"] = []
    for i, loss_tens in enumerate(nn_nodes["supervised"]["extra_losses"]["xuv_individual_coef_loss"]):
        # linear, 2nd, 3rd, 4th , 5th as list
        tf_loggers["individual"]["xuv_coefs_test"].append(tf.summary.scalar("xuv_coef{}_test".format(str(i+1)), loss_tens))

    # logger for coefficients in training data
    tf_loggers["individual"]["xuv_coefs_train"] = []
    for i, loss_tens in enumerate(nn_nodes["supervised"]["extra_losses"]["xuv_individual_coef_loss"]):
        # linear, 2nd, 3rd, 4th , 5th as list
        tf_loggers["individual"]["xuv_coefs_train"].append(tf.summary.scalar("xuv_coef{}_train".format(str(i+1)), loss_tens))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # log for IR parameters individually in test and train data
    tf_loggers["individual"]["ir_params_test"] = {}
    for key in nn_nodes["supervised"]["extra_losses"]["ir_loss_individual"].keys():
        tf_loggers["individual"]["ir_params_test"][key] = tf.summary.scalar("IR_"+key+"_test", nn_nodes["supervised"]["extra_losses"]["ir_loss_individual"][key])

    tf_loggers["individual"]["ir_params_train"] = {}
    for key in nn_nodes["supervised"]["extra_losses"]["ir_loss_individual"].keys():
        tf_loggers["individual"]["ir_params_train"][key] = tf.summary.scalar("IR_"+key+"_train", nn_nodes["supervised"]["extra_losses"]["ir_loss_individual"][key])

    return tf_loggers

def create_sample_plot(samples_per_plot=3):
    fig = plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.05,
                            wspace=0.2, hspace=0.1)
    gs = fig.add_gridspec(4,int(samples_per_plot*2))

    plot_rows = []
    for i in range(samples_per_plot):

        column_axes = {}

        column_axes['actual_ir'] = fig.add_subplot(gs[0, 2*i])
        column_axes['actual_xuv'] = fig.add_subplot(gs[0, 2*i+1])
        column_axes['actual_xuv_twinx'] = column_axes['actual_xuv'].twinx()

        column_axes['input_trace'] = fig.add_subplot(gs[1, 2*i:2*i+2])

        column_axes['predict_ir'] = fig.add_subplot(gs[2, 2*i])
        column_axes['predict_xuv'] = fig.add_subplot(gs[2, 2*i+1])
        column_axes['predict_xuv_twinx'] = column_axes['predict_xuv'].twinx()

        column_axes['reconstruct'] = fig.add_subplot(gs[3, 2*i:2*i+2])

        plot_rows.append(column_axes)

    return plot_rows, fig

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')

def conv2d_nopad(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding="VALID")

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init_bias_vals)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

def multires_layer(input, input_channels, filter_sizes, stride=1):
    # list of layers
    filters = []
    for filter_size in filter_sizes:
        # create filter
        filters.append(convolutional_layer(input, shape=[filter_size, filter_size,
                        input_channels, input_channels], activate='leaky', stride=[stride, stride]))

    concat_layer = tf.concat(filters, axis=3)
    return concat_layer

def convolutional_layer(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b

def convolutional_layer_nopadding(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d_nopad(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d_nopad(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d_nopad(input_x, W, stride) + b

def max_pooling_layer(input_x, pool_size_val,  stride_val, pad=False):
    if pad:
        return tf.layers.max_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="SAME")
    else:
        return tf.layers.max_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="VALID")

def avg_pooling_layer(input_x, pool_size_val,  stride_val, pad=False):
    if pad:
        return tf.layers.average_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="SAME")
    else:
        return tf.layers.average_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="VALID")

def gan_network(input):
    xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs
    output_length = xuv_phase_coefs - 1 + 4     # remove 1 for no linear phase...
                                                # add 4 for ir params


    with tf.variable_scope("gan"):
        hidden1 = tf.layers.dense(inputs=input, units=128)

        alpha = 0.01
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        hidden2 = tf.layers.dense(inputs=hidden1, units=128)

        hidden2 = tf.maximum(alpha * hidden2, hidden2)

        # output of neural net between -1 and 1
        output = tf.layers.dense(hidden2, units=output_length, activation=tf.nn.tanh)

        # scaling factor between 0 and 1
        s_out = tf.layers.dense(hidden2, units=1, activation=tf.nn.sigmoid)

        # output : [---xuv_coefs-- --ir_params--]
        # represent taylor series coefficients
        xuv_out = output[:, 0:xuv_phase_coefs - 1]
        ir_out = output[:, xuv_phase_coefs - 1:]

        # append 0 to the xuv output because 0 linear phase
        samples_in = tf.shape(xuv_out)[0]
        zeros_vec = tf.fill([samples_in, 1], 0.0)
        xuv_out_nolin = tf.concat([zeros_vec, xuv_out], axis=1)

        # normalize the xuv output
        summation = tf.reduce_sum(tf.abs(xuv_out_nolin), axis=1)
        quotient = (1 / s_out) * summation
        xuv_coefs_normalized = xuv_out_nolin / quotient

        # create label with coefs and params
        coefs_params_label = tf.concat([xuv_coefs_normalized, ir_out], axis=1)

        # generate complex fields from these coefs
        xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_normalized)
        ir_E_prop = tf_functions.ir_from_params(ir_out)["E_prop"]

        # concat these vectors to make a label
        xuv_ir_field_label = concat_fields(xuv=xuv_E_prop["f_cropped"], ir=ir_E_prop["f_cropped"])

        outputs = {}
        outputs["xuv_ir_field_label"] = xuv_ir_field_label
        outputs["coefs_params_label"] = coefs_params_label
        outputs["ir_E_prop"] = ir_E_prop
        outputs["xuv_E_prop"] = xuv_E_prop


        return outputs

def part1_purple(input):
    conv1 = convolutional_layer_nopadding(input, shape=[21, 8, 1, 20], activate='relu', stride=[1, 1])
    # print("conv1", conv1)

    pool1 = max_pooling_layer(conv1, pool_size_val=[13, 5], stride_val=[3, 3])
    # print("pool1", pool1)

    conv2 = convolutional_layer_nopadding(pool1, shape=[13, 5, 20, 40], activate='relu', stride=[1, 1])
    # print("conv2", conv2)

    pool2 = max_pooling_layer(conv2, pool_size_val=[9, 3], stride_val=[2, 2])

    return pool2

def part2_grey(input):
    # center
    conv31 = convolutional_layer_nopadding(input, shape=[13, 5, 40, 40], activate='relu', stride=[1, 1])
    conv322 = convolutional_layer_nopadding(conv31, shape=[13, 4, 40, 20], activate='relu', stride=[1, 1])

    #left
    pool3 = max_pooling_layer(input, pool_size_val=[14, 3], stride_val=[2, 2])
    conv321 = convolutional_layer_nopadding(pool3, shape=[1, 1, 40, 20], activate='relu', stride=[1, 1])

    #right
    conv323 = convolutional_layer_nopadding(input, shape=[25, 8, 40, 20], activate='relu', stride=[1, 1])

    conc1 = tf.concat([conv321, conv322, conv323], axis=3)

    return conc1

def part3_green(input):
    # left side
    conv4 = convolutional_layer_nopadding(input, shape=[3, 2, 60, 40], activate='relu', stride=[1, 1])

    # right side
    pool4 = max_pooling_layer(input, pool_size_val=[3, 2], stride_val=[1, 1])

    # concatinate
    conc2 = tf.concat([conv4, pool4], axis=3)

    return conc2

def noise_resistant_phase_retrieval_net(input):
    K_values = phase_parameters.params.K
    tau_values = phase_parameters.params.delay_values
    xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs
    total_coefs_params_length = int(xuv_phase_coefs + 4)
    with tf.variable_scope("phase"):
        x_image = tf.reshape(input, [-1, len(K_values), len(tau_values), 1])

        print("np.shape(x_image)", np.shape(x_image))
        # x_image = tf.placeholder(shape=(None, 64, 64, 1), dtype=tf.float32)

        pool2 = part1_purple(x_image)

        conc1 = part2_grey(pool2)

        conc2 = part3_green(conc1)

        pool51 = avg_pooling_layer(conc2, pool_size_val=[3, 3], stride_val=[1, 1])
        # print("pool51", pool51)

        pool52 = avg_pooling_layer(pool2, pool_size_val=[5, 5], stride_val=[5, 5], pad=True)
        # print("pool52", pool52)

        pool53 = avg_pooling_layer(conc1, pool_size_val=[3, 3], stride_val=[2, 2])
        # print("pool53", pool53)

        pool51_flat = tf.contrib.layers.flatten(pool51)
        pool52_flat = tf.contrib.layers.flatten(pool52)
        pool53_flat = tf.contrib.layers.flatten(pool53)

        conc3 = tf.concat([pool51_flat, pool52_flat, pool53_flat], axis=1)

        fc5 = tf.layers.dense(inputs=conc3, units=256)

        # dropout
        hold_prob = tf.placeholder_with_default(1.0, shape=())
        dropout_layer = tf.nn.dropout(fc5, keep_prob=hold_prob)

        # output layer
        predicted_coefficients_params = normal_full_layer(dropout_layer, total_coefs_params_length)
        xuv_coefs_pred = tf.placeholder_with_default(predicted_coefficients_params[:, 0:phase_parameters.params.xuv_phase_coefs], shape=[None, 5])
        ir_params_pred = predicted_coefficients_params[:, phase_parameters.params.xuv_phase_coefs:]

        # generate fields from coefficients
        xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_pred)
        ir_E_prop = tf_functions.ir_from_params(ir_params_pred)["E_prop"]

        # generate a label from the complex fields
        xuv_ir_field_label = concat_fields(xuv=xuv_E_prop["f_cropped"], ir=ir_E_prop["f_cropped"])


        phase_net_output = {}
        phase_net_output["xuv_ir_field_label"] = xuv_ir_field_label
        phase_net_output["ir_E_prop"] = ir_E_prop
        phase_net_output["xuv_E_prop"] = xuv_E_prop
        phase_net_output["predicted_coefficients_params"] = predicted_coefficients_params

        return phase_net_output, hold_prob, xuv_coefs_pred, ir_params_pred

def phase_retrieval_net(input):
    K_values = phase_parameters.params.K
    tau_values = phase_parameters.params.delay_values

    xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs
    total_coefs_params_length = int(xuv_phase_coefs + 4)

    # define phase retrieval neural network
    with tf.variable_scope("phase"):

        # # dense network
        # x_image_flat = tf.reshape(input, [-1, len(K_values)*len(tau_values)])
        # dense1 = normal_full_layer(x_image_flat, 1024)
        # # dense layer 1
        # full_layer_one = normal_full_layer(dense1, 1024)


        # convolutional layers
        # # shape = [sizex, sizey, channels, filters/features]
        # convo_1 = convolutional_layer(x_image, shape=[4, 4, 1, 32], activate='none', stride=[2, 2])
        # convo_2 = convolutional_layer(convo_1, shape=[2, 2, 32, 32], activate='none', stride=[2, 2])
        # convo_3 = convolutional_layer(convo_2, shape=[1, 1, 32, 32], activate='leaky', stride=[1, 1])


        # convolutional network
        # input image
        x_image = tf.reshape(input, [-1, len(K_values), len(tau_values), 1])

        # six convolutional layers
        multires_filters = [11, 7, 5, 3]

        multires_layer_1 = multires_layer(input=x_image, input_channels=1, filter_sizes=multires_filters, stride=2)

        # use this to pass output forward
        # fwd_1 = tf.contrib.layers.flatten(multires_layer_1)

        multires_layer_2 = multires_layer(input=multires_layer_1, input_channels=4,
                                          filter_sizes=multires_filters, stride=2)

        multires_layer_3 = multires_layer(input=multires_layer_2, input_channels=16,
                                          filter_sizes=multires_filters, stride=2)

        convo_3_flat = tf.contrib.layers.flatten(multires_layer_3)
        full_layer_one = normal_full_layer(convo_3_flat, 512)
        #full_layer_one = normal_full_layer(convo_3_flat, 2)
        #print("layer needs to be set to 1024!!")

        # dropout
        hold_prob = tf.placeholder_with_default(1.0, shape=())
        dropout_layer = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

        # neural net output coefficients
        predicted_coefficients_params = normal_full_layer(dropout_layer, total_coefs_params_length)
        # predicted_coefficients_params = normal_full_layer(dropout_layer, total_coefs_params_length)

        xuv_coefs_pred = tf.placeholder_with_default(predicted_coefficients_params[:, 0:phase_parameters.params.xuv_phase_coefs], shape=[None, 5])
        ir_params_pred = predicted_coefficients_params[:, phase_parameters.params.xuv_phase_coefs:]

        # generate fields from coefficients
        xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_pred)
        ir_E_prop = tf_functions.ir_from_params(ir_params_pred)["E_prop"]

        # generate a label from the complex fields
        xuv_ir_field_label = concat_fields(xuv=xuv_E_prop["f_cropped"], ir=ir_E_prop["f_cropped"])


        phase_net_output = {}
        phase_net_output["xuv_ir_field_label"] = xuv_ir_field_label
        phase_net_output["ir_E_prop"] = ir_E_prop
        phase_net_output["xuv_E_prop"] = xuv_E_prop
        phase_net_output["predicted_coefficients_params"] = predicted_coefficients_params

        return phase_net_output, hold_prob, xuv_coefs_pred, ir_params_pred

def setup_neural_net():
    K_values = phase_parameters.params.K
    tau_values = phase_parameters.params.delay_values

    xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs

    print('Setting up multires layer network with more conv weights')

    # define the label for supervised learning of phase retrieval net
    total_coefs_params_length = int(xuv_phase_coefs + 4)

    # define GAN network
    gan_input = tf.placeholder(tf.float32, shape=[1, 100])

    # GAN output is used to create XUV field and streaking trace
    gan_output = gan_network(input=gan_input)

    # use the fields to generate streaking trace
    # sample size of one required as of now
    x = tf_functions.streaking_trace(xuv_cropped_f_in=gan_output["xuv_E_prop"]["f_cropped"][0],
                                        ir_cropped_f_in=gan_output["ir_E_prop"]["f_cropped"][0])
    x_flat = tf.reshape(x, [1, -1])
    # this placeholder accepts either an input as placeholder (supervised learning)
    # or it will default to the GAN generated fields as input
    x_in = tf.placeholder_with_default(x_flat, shape=(None, int(len(K_values) * len(tau_values))))


    phase_net_output, hold_prob, xuv_coefs_pred, ir_params_pred = noise_resistant_phase_retrieval_net(input=x_in)


    # pass image through phase retrieval network
    # phase_net_output, hold_prob, xuv_coefs_pred, ir_params_pred = phase_retrieval_net(input=x_in)


    # create label for supervised learning
    actual_coefs_params = tf.placeholder(tf.float32, shape=[None, total_coefs_params_length])
    supervised_label_fields = create_fields_label_from_coefs_params(actual_coefs_params)

    # generate the reconstructed trace
    reconstructed_trace = tf_functions.streaking_trace(
                    xuv_cropped_f_in=phase_net_output["xuv_E_prop"]["f_cropped"][0],
                    ir_cropped_f_in=phase_net_output["ir_E_prop"]["f_cropped"][0])

    # generate proof trace
    reconstructed_proof = tf_functions.proof_trace(reconstructed_trace)
    # input proof trace
    x_in_reshaped = tf.reshape(x_in, [len(K_values), len(tau_values)])
    input_image_proof = tf_functions.proof_trace(x_in_reshaped)


    # generate autocorrelation trace
    reconstructed_autocorrelate = tf_functions.autocorrelate(reconstructed_trace)
    # input autocorrelation trace
    input_image_autocorrelate = tf_functions.autocorrelate(x_in_reshaped)


    # divide the variables to train with gan and phase retrieval net individually
    tvars = tf.trainable_variables()
    phase_net_vars = [var for var in tvars if "phase" in var.name]
    gan_net_vars = [var for var in tvars if "gan" in var.name]


    #........................................................
    #........................................................
    # ..............define loss functions....................
    #........................................................
    #........................................................




    # #........................................................
    # # .............GAN NETWORK LOSS FUNCTIONS................
    # #........................................................
    # # maximize loss between complex fields
    # gan_fields_loss = (1/tf.losses.mean_squared_error(labels=gan_output["xuv_ir_field_label"],
    #                                                    predictions=phase_net_output["xuv_ir_field_label"]))
    # gan_LR = tf.placeholder(tf.float32, shape=[])
    # gan_optimizer = tf.train.AdamOptimizer(learning_rate=gan_LR)
    # gan_network_train = gan_optimizer.minimize(gan_fields_loss, var_list=gan_net_vars)




    # ........................................................
    # ........SUPERVISED LEARNING LOSS FUNCTIONS..............
    # ........................................................
    s_LR = tf.placeholder(tf.float32, shape=[])

    # phase curve loss function
    phase_network_phasecurve_loss = tf.losses.mean_squared_error(
                            labels=supervised_label_fields["xuv_E_prop"]["phasecurve_cropped"],
                            predictions=phase_net_output["xuv_E_prop"]["phasecurve_cropped"])
    phase_phasecurve_optimizer = tf.train.AdamOptimizer(learning_rate=s_LR)
    phase_network_train_phasecurve = phase_phasecurve_optimizer.minimize(
                            phase_network_phasecurve_loss, var_list=phase_net_vars)

    # fields loss function for training phase retrieval network
    phase_network_fields_loss = tf.losses.mean_squared_error(
                            labels=supervised_label_fields["xuv_ir_field_label"],
                            predictions=phase_net_output["xuv_ir_field_label"])
    phase_fields_optimizer = tf.train.AdamOptimizer(learning_rate=s_LR)
    phase_network_train_fields = phase_fields_optimizer.minimize(
                            phase_network_fields_loss, var_list=phase_net_vars)

    # modify scaler to phase coefficients here
    # supervised_label_fields["actual_coefs_params"]
    # phase_net_output["predicted_coefficients_params"]

    # coefs and params loss function for training phase retrieval network
    # original loss function
    phase_network_coefs_params_loss = tf.losses.mean_squared_error(
                            labels=supervised_label_fields["actual_coefs_params"],
                            predictions=phase_net_output["predicted_coefficients_params"])

    # construct loss function with individual cosfficients
    xuv_coef_loss_w = tf.losses.mean_squared_error(
                            labels=supervised_label_fields["xuv_coefs_actual"][:,1:],
                            predictions=xuv_coefs_pred[:,1:])

    # construct a vector of for the IR loss with only the intensity and phaseshift
    ir_param_loss_w = tf.losses.mean_squared_error(
                            labels=convert_ir_params(supervised_label_fields["ir_params_actual"]),
                            predictions=convert_ir_params(ir_params_pred))
    phase_network_coefs_params_loss_individual = xuv_coef_loss_w + ir_param_loss_w

    # original
    # phase_coefs_params_optimizer = tf.train.AdamOptimizer(learning_rate=s_LR)
    # phase_network_train_coefs_params = phase_coefs_params_optimizer.minimize(
    #                         phase_network_coefs_params_loss, var_list=phase_net_vars)

    # individual
    phase_coefs_params_optimizer = tf.train.AdamOptimizer(learning_rate=s_LR)
    phase_network_train_coefs_params = phase_coefs_params_optimizer.minimize(
                            phase_network_coefs_params_loss_individual, var_list=phase_net_vars)


    # =========================================================================
    # define individual xuv / ir /xuc coefficient loss functions to view errors
    # =========================================================================
    xuv_loss = tf.losses.mean_squared_error(
            labels=supervised_label_fields["xuv_coefs_actual"],
            predictions=xuv_coefs_pred)

    ir_loss = tf.losses.mean_squared_error(
            labels=supervised_label_fields["ir_params_actual"],
            predictions=ir_params_pred)

    xuv_individual_coef_loss = []
    # linear, 2nd order, 3rd, 4th, 5th etc...
    for i in range(int(xuv_coefs_pred.get_shape()[1])):
        xuv_coef_loss = tf.losses.mean_squared_error(
                labels=supervised_label_fields["xuv_coefs_actual"][:,i],
                predictions=xuv_coefs_pred[:,i])
        xuv_individual_coef_loss.append(xuv_coef_loss)

    ir_loss_individual = {}
    for i, key in enumerate(["phase", "clambda", "pulseduration", "I"]):
        ir_param_loss = tf.losses.mean_squared_error(
                labels=supervised_label_fields["ir_params_actual"][:,i],
                predictions=ir_params_pred[:,i])

        # add extra term for the cos of phase term
        if key == "phase":

            phase_pred = tf_functions.ir_from_params(ir_params_pred)["scaled_values"]["phase"]
            phase_true = tf_functions.ir_from_params(supervised_label_fields["ir_params_actual"])["scaled_values"]["phase"]

            # ir_loss_individual["phase_cos_rad"] = tf.losses.mean_squared_error(
                # labels=tf.cos(phase_true)+tf.sin(phase_true),
                # predictions=tf.cos(phase_pred)+tf.sin(phase_pred))

            ir_loss_individual["phase_cos"] = tf.losses.mean_squared_error(
                labels=tf.cos(phase_true),
                predictions=tf.cos(phase_pred))

            ir_loss_individual["phase_sin"] = tf.losses.mean_squared_error(
                labels=tf.sin(phase_true),
                predictions=tf.sin(phase_pred))

            # this is the old cost function that doesnt make any sense
            # ir_loss_individual["phase_cos_old"] = tf.losses.mean_squared_error(
                # labels=tf.cos(supervised_label_fields["ir_params_actual"][:,i]),
                # predictions=tf.cos(ir_params_pred[:,i]))

        ir_loss_individual[key] = ir_param_loss



    # ..........................................................
    # .........UNSUPERVISED LEARNING LOSS FUNCTION..............
    # ..........................................................
    u_LR = tf.placeholder(tf.float32, shape=[])

    # regular cost function
    unsupervised_learning_loss = tf.losses.mean_squared_error(labels=x_in,
                            predictions=tf.reshape(reconstructed_trace, [1, -1]))
    unsupervised_optimizer = tf.train.AdamOptimizer(learning_rate=u_LR)
    unsupervised_train = unsupervised_optimizer.minimize(unsupervised_learning_loss,
                                                        var_list=phase_net_vars)

    # # log cost function
    # # log1 = log_base(x=0.5, base=10.0, translate=1)
    # u_base = tf.placeholder(tf.float32, shape=[])
    # u_translate = tf.placeholder(tf.float32, shape=[])
    # unsupervised_learning_loss_log = tf.losses.mean_squared_error(
    #                         labels=log_base(x=x_in, base=u_base, translate=u_translate),
    #                         predictions=log_base(x=tf.reshape(reconstructed_trace, [1, -1]),
    #                                              base=u_base,
    #                                              translate=u_translate)
    # )
    # unsupervised_optimizer_log = tf.train.AdamOptimizer(learning_rate=u_LR)
    # unsupervised_train_log = unsupervised_optimizer_log.minimize(unsupervised_learning_loss_log,
    #                                                     var_list=phase_net_vars)


    # ..........................................................
    # .................PROOF RETRIEVAL LOSS FUNC................
    # ..........................................................
    # regular cost function
    proof_unsupervised_learning_loss = tf.losses.mean_squared_error(
                            labels=tf.reshape(input_image_proof["proof"], [1, -1]),
                            predictions=tf.reshape(reconstructed_proof["proof"], [1, -1]))
    proof_unsupervised_optimizer = tf.train.AdamOptimizer(learning_rate=u_LR)
    proof_unsupervised_train = proof_unsupervised_optimizer.minimize(
                            proof_unsupervised_learning_loss,
                            var_list=phase_net_vars)

    # ..........................................................
    # .............AUTOCORRELATION RETRIEVAL LOSS FUNC..........
    # ..........................................................
    # regular cost function
    autocorrelate_unsupervised_learning_loss = tf.losses.mean_squared_error(
        labels=tf.reshape(input_image_autocorrelate, [1, -1]),
        predictions=tf.reshape(reconstructed_autocorrelate, [1, -1]))
    autocorrelate_unsupervised_optimizer = tf.train.AdamOptimizer(learning_rate=u_LR)
    autocorrelate_unsupervised_train = autocorrelate_unsupervised_optimizer.minimize(
                                            autocorrelate_unsupervised_learning_loss,
                                            var_list=phase_net_vars)

    # +++++++++++++++++++++++++++++++++++++
    # ++++++++++BOOTSTRAP METHOD+++++++++++
    # +++++++++++++++++++++++++++++++++++++
    norm_bootstrap_loss, norm_bootstrap_train, norm_bootstrap_indexes_ph = bootstrap(
                    recons_trace=reconstructed_trace, input_trace=x_in,
                    learning_rate_in=u_LR, train_variables=phase_net_vars
    )

    proof_bootstrap_loss, proof_bootstrap_train, proof_bootstrap_indexes_ph = bootstrap(
                    recons_trace=reconstructed_proof["proof"], input_trace=input_image_proof["proof"],
                    learning_rate_in=u_LR, train_variables=phase_net_vars
    )

    auto_bootstrap_loss, auto_bootstrap_train, auto_bootstrap_indexes_ph = bootstrap(
                    recons_trace=reconstructed_autocorrelate, input_trace=input_image_autocorrelate,
                    learning_rate_in=u_LR, train_variables=phase_net_vars
    )



    # ..........................................................
    # ...................DEFINE NODES FOR USE...................
    # ..........................................................

    nn_nodes = {}
    nn_nodes["gan"] = {}
    nn_nodes["supervised"] = {}
    nn_nodes["unsupervised"] = {}
    nn_nodes["general"] = {}

    # nn_nodes["gan"]["gan_input"] = gan_input
    # nn_nodes["gan"]["gan_output"] = gan_output
    # nn_nodes["gan"]["gan_LR"] = gan_LR
    # nn_nodes["gan"]["gan_network_train"] = gan_network_train

    nn_nodes["supervised"]["x_in"] = x_in
    nn_nodes["supervised"]["actual_coefs_params"] = actual_coefs_params
    nn_nodes["supervised"]["phase_network_train_phasecurve"] = phase_network_train_phasecurve
    nn_nodes["supervised"]["phase_network_train_fields"] = phase_network_train_fields
    nn_nodes["supervised"]["phase_network_train_coefs_params"] = phase_network_train_coefs_params
    nn_nodes["supervised"]["s_LR"] = s_LR
    nn_nodes["supervised"]["phase_network_phasecurve_loss"] = phase_network_phasecurve_loss
    nn_nodes["supervised"]["phase_network_fields_loss"] = phase_network_fields_loss
    nn_nodes["supervised"]["phase_network_coefs_params_loss"] = phase_network_coefs_params_loss
    nn_nodes["supervised"]["supervised_label_fields"] = supervised_label_fields
    # avg ir and xuv loss / individual xuv coefficient loss functions
    nn_nodes["supervised"]["extra_losses"] = {}
    nn_nodes["supervised"]["extra_losses"]["ir_loss"] = ir_loss
    nn_nodes["supervised"]["extra_losses"]["ir_loss_individual"] = ir_loss_individual
    nn_nodes["supervised"]["extra_losses"]["xuv_loss"] = xuv_loss
    nn_nodes["supervised"]["extra_losses"]["xuv_individual_coef_loss"] = xuv_individual_coef_loss



    nn_nodes["unsupervised"]["x_in"] = x_in
    nn_nodes["unsupervised"]["unsupervised_train"] = unsupervised_train
    # nn_nodes["unsupervised"]["unsupervised_train_log"] = unsupervised_train_log
    nn_nodes["unsupervised"]["u_LR"] = u_LR
    nn_nodes["unsupervised"]["unsupervised_learning_loss"] = unsupervised_learning_loss
    # nn_nodes["unsupervised"]["unsupervised_learning_loss_log"] = unsupervised_learning_loss_log
    # nn_nodes["unsupervised"]["u_base"] = u_base
    # nn_nodes["unsupervised"]["u_translate"] = u_translate

    nn_nodes["unsupervised"]["proof"] = {}
    nn_nodes["unsupervised"]["proof"]["x_in"] = x_in
    nn_nodes["unsupervised"]["proof"]["u_LR"] = u_LR
    nn_nodes["unsupervised"]["proof"]["reconstructed_proof"] = reconstructed_proof
    nn_nodes["unsupervised"]["proof"]["input_image_proof"] = input_image_proof
    nn_nodes["unsupervised"]["proof"]["proof_unsupervised_train"] = proof_unsupervised_train
    nn_nodes["unsupervised"]["proof"]["proof_unsupervised_learning_loss"] = proof_unsupervised_learning_loss

    nn_nodes["unsupervised"]["autocorrelate"] = {}
    nn_nodes["unsupervised"]["autocorrelate"]["x_in"] = x_in
    nn_nodes["unsupervised"]["autocorrelate"]["u_LR"] = u_LR
    nn_nodes["unsupervised"]["autocorrelate"]["reconstructed_autocorrelate"] = reconstructed_autocorrelate
    nn_nodes["unsupervised"]["autocorrelate"]["input_image_autocorrelate"] = input_image_autocorrelate
    nn_nodes["unsupervised"]["autocorrelate"]["autocorrelate_unsupervised_train"] = autocorrelate_unsupervised_train
    nn_nodes["unsupervised"]["autocorrelate"]["autocorrelate_unsupervised_learning_loss"] = autocorrelate_unsupervised_learning_loss


    # add nodes for bootstrap method
    nn_nodes["unsupervised"]["bootstrap"] = {}
    
    nn_nodes["unsupervised"]["bootstrap"]["u_LR"] = u_LR

    nn_nodes["unsupervised"]["bootstrap"]["normal"] = {}
    nn_nodes["unsupervised"]["bootstrap"]["normal"]["loss"] = norm_bootstrap_loss
    nn_nodes["unsupervised"]["bootstrap"]["normal"]["train"] = norm_bootstrap_train
    nn_nodes["unsupervised"]["bootstrap"]["normal"]["indexes_ph"] = norm_bootstrap_indexes_ph

    nn_nodes["unsupervised"]["bootstrap"]["proof"] = {}
    nn_nodes["unsupervised"]["bootstrap"]["proof"]["loss"] = proof_bootstrap_loss
    nn_nodes["unsupervised"]["bootstrap"]["proof"]["train"] = proof_bootstrap_train
    nn_nodes["unsupervised"]["bootstrap"]["proof"]["indexes_ph"] = proof_bootstrap_indexes_ph

    nn_nodes["unsupervised"]["bootstrap"]["auto"] = {}
    nn_nodes["unsupervised"]["bootstrap"]["auto"]["loss"]  = auto_bootstrap_loss
    nn_nodes["unsupervised"]["bootstrap"]["auto"]["train"] = auto_bootstrap_train
    nn_nodes["unsupervised"]["bootstrap"]["auto"]["indexes_ph"]  = auto_bootstrap_indexes_ph


    nn_nodes["general"]["phase_net_output"] = phase_net_output
    nn_nodes["general"]["reconstructed_trace"] = reconstructed_trace
    nn_nodes["general"]["hold_prob"] = hold_prob
    nn_nodes["general"]["x_in"] = x_in
    nn_nodes["general"]["xuv_coefs_pred"] = xuv_coefs_pred

    return nn_nodes

def bootstrap(recons_trace, input_trace, learning_rate_in, train_variables):
    bootstrap_loss, bootstrap_indexes_ph = calc_bootstrap_error(recons_trace, input_trace)
    bootstrap_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_in)
    bootstrap_train = bootstrap_optimizer.minimize(
        bootstrap_loss , var_list=train_variables
    )
    return bootstrap_loss, bootstrap_train, bootstrap_indexes_ph

def calc_bootstrap_error(recons_trace_in, input_trace_in):
    recons_vec = tf.reshape(recons_trace_in, [-1])
    input_vec = tf.reshape(input_trace_in, [-1])
    # use 2/3rds of the points
    number_of_bootstrap_indexes = int( (2/3) * recons_vec.get_shape().as_list()[0])
    bootstrap_indexes_ph = tf.placeholder(tf.int32, shape=[number_of_bootstrap_indexes])
    recons_values = tf.gather(recons_vec, bootstrap_indexes_ph)
    input_values = tf.gather(input_vec, bootstrap_indexes_ph)
    # mean squared error
    bootstrap_loss = tf.losses.mean_squared_error(
        labels=input_values, predictions=recons_values
    )
    return bootstrap_loss, bootstrap_indexes_ph

def take_norm_angle(angle_in):
    cnum = tf.exp(tf.complex(imag=angle_in, real=tf.zeros_like(angle_in)))
    angle_norm = tf.math.angle(cnum)
    return angle_norm

def divide_to_cos_sin(angle_in):

    cos_angle = tf.cos(angle_in)
    sin_angle = tf.sin(angle_in)

    return cos_angle, sin_angle




if __name__ == "__main__":
    phase_net_train = PhaseNetTrain(modelname=sys.argv[1])
    # phase_net_train = PhaseNetTrain(modelname='test_test')
    phase_net_train.supervised_learn()

