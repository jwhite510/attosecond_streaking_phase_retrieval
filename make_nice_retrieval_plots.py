import tensorflow as tf
import crab_tf2
xuv_time_domain_func = crab_tf2.xuv_time_domain
import network2
import unsupervised
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc


def create_plots():

    fig = plt.figure(figsize=(10,6.5))
    gs = fig.add_gridspec(2,4)

    axes = {}

    axes['input_image'] = fig.add_subplot(gs[0,0:3])
    axes['input_xuv_time'] = fig.add_subplot(gs[0,3])
    # axes['input_xuv'] = fig.add_subplot(gs[0,1])
    # axes['input_ir'] = fig.add_subplot(gs[0,0])

    axes['generated_image'] = fig.add_subplot(gs[1,0:3])
    axes['predicted_xuv_time'] = fig.add_subplot(gs[1, 3])
    # axes['predicted_xuv'] = fig.add_subplot(gs[2, 1])
    # axes['predicted_ir'] = fig.add_subplot(gs[2, 0])

    # add time plots

    return axes



def update_plots(generated_image, input_image, actual_fields, predicted_fields):

    # set phase angle to 0
    # find angle 0
    index_0_angle = 25
    #plt.figure(999)
    #plt.plot(np.real(actual_fields['xuv_f']), color='blue')
    #plt.plot(np.imag(actual_fields['xuv_f']), color='red')
    #plt.plot(np.unwrap(np.angle(actual_fields['xuv_f'])), color='green')
    #plt.plot([index_0_angle,index_0_angle], [-1, 1])
    #plt.ioff()
    #plt.show()
    xuv_actual_time = sess.run(xuv_time_domain_func, feed_dict={crab_tf2.xuv_cropped_f: actual_fields['xuv_f']})
    xuv_predicted_time = sess.run(xuv_time_domain_func, feed_dict={crab_tf2.xuv_cropped_f: predicted_fields['xuv_f']})


    # set phase angle to 0
    predicted_fields['xuv_f'] = predicted_fields['xuv_f'] * np.exp(-1j * np.angle(predicted_fields['xuv_f'][index_0_angle]))

    iteration = i+1

    # calculate loss
    loss_value = sess.run(network2.u_losses, feed_dict={network2.x: trace})
    print('iteration: {}'.format(iteration))
    print('loss_value: ', loss_value)

    xuv_t_vals_si = crab_tf2.xuv.tmat * sc.physical_constants['atomic unit of time'][0]
    axes['input_xuv_time'].cla()
    axes['input_xuv_time'].text(0.0, 1.05, 'Actual XUV E(t)', transform=axes['input_xuv_time'].transAxes, backgroundcolor='white')
    axes['input_xuv_time'].plot(xuv_t_vals_si*1e18, np.real(xuv_actual_time), color='blue', alpha=0.5)
    axes['input_xuv_time'].plot(xuv_t_vals_si*1e18, np.abs(xuv_actual_time), color='black')
    axes['input_xuv_time'].set_xlabel("time [as]")
    axes['input_xuv_time'].set_ylabel("Electric Field [arbitrary units]")
    axes['input_xuv_time'].yaxis.tick_right()
    axes['input_xuv_time'].yaxis.set_label_position("right")
    axes['input_xuv_time'].set_yticks([0.02, 0.01, 0, -0.01, -0.02])


    axes['input_image'].cla()
    tauvalues_si = crab_tf2.tau_values * sc.physical_constants['atomic unit of time'][0]
    axes['input_image'].pcolormesh(tauvalues_si*1e15, crab_tf2.p_values, input_image, cmap='jet')
    axes['input_image'].text(0.0, 1.05, 'Input Streaking Trace', transform=axes['input_image'].transAxes, backgroundcolor='white')
    axes['input_image'].set_ylabel("momentum [atomic units]")
    axes['input_image'].set_xlabel("delay [fs]")


    axes['predicted_xuv_time'].cla()
    axes['predicted_xuv_time'].text(0.0, 1.05, 'Predicted XUV E(t)', transform=axes['predicted_xuv_time'].transAxes,
                                backgroundcolor='white')
    axes['predicted_xuv_time'].plot(xuv_t_vals_si*1e18, np.real(xuv_predicted_time), color='blue', alpha=0.5, label='Real E(t)')
    axes['predicted_xuv_time'].plot(xuv_t_vals_si*1e18, np.abs(xuv_predicted_time), color='black', label='|E(t)|')
    axes['predicted_xuv_time'].set_xlabel("time [as]")
    axes['predicted_xuv_time'].set_ylabel("Electric Field [arbitrary units]")
    axes['predicted_xuv_time'].legend(bbox_to_anchor=(0.5, -0.4), ncol=2, loc='center')
    #axes['predicted_xuv_time'].legend(loc=4)
    axes['predicted_xuv_time'].yaxis.tick_right()
    axes['predicted_xuv_time'].yaxis.set_label_position("right")
    axes['predicted_xuv_time'].set_yticks([0.02, 0.01, 0, -0.01, -0.02])



    axes['generated_image'].cla()
    axes['generated_image'].pcolormesh(tauvalues_si*1e15, crab_tf2.p_values, generated_image, cmap='jet')
    axes['generated_image'].text(0.0, 1.05, 'Generated Streaking Trace', transform=axes['generated_image'].transAxes, backgroundcolor='white')
    #axes['generated_image'].text(0.0, 1.0, 'iteration: {}'.format(iteration), transform = axes['generated_image'].transAxes, backgroundcolor='white')

    axes['generated_image'].text(0.1, 0.1, 'MSE: {}'.format(str(loss_value)),
                                 transform=axes['generated_image'].transAxes, backgroundcolor='white')
    axes['generated_image'].set_ylabel("momentum [atomic units]")
    axes['generated_image'].set_xlabel("delay [fs]")

    # save image
    # dir = "./nnpictures/unsupervised/" + modelname + "/" + run_name + "/"
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)
    # plt.savefig(dir + str(iteration) + ".png")


    plt.subplots_adjust(left=0.08, right=0.92, hspace=0.4, wspace=0.3, bottom=0.18)
    plt.show()




def generate_images_and_plot():
    # generate an image from the input image
    generated_image = sess.run(network2.image, feed_dict={network2.x: trace})
    trace_2d = trace.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values))
    predicted_fields_vector = sess.run(network2.y_pred, feed_dict={network2.x: trace})

    predicted_fields = {}
    predicted_fields['xuv_f'], predicted_fields['ir_f'] = network2.separate_xuv_ir_vec(predicted_fields_vector[0])

    # plot the trace and generated image
    update_plots(generated_image, trace_2d, actual_fields, predicted_fields)


def create_sample_plot(samples_per_plot=3):

    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                            wspace=0.5, hspace=0.4)
    gs = fig.add_gridspec(4,int(samples_per_plot*2))

    plot_rows = []
    for i in range(samples_per_plot):

        column_axes = {}

        column_axes['actual_ir'] = fig.add_subplot(gs[0, 2*i])
        column_axes['actual_xuv'] = fig.add_subplot(gs[0, 2*i+1])

        column_axes['input_trace'] = fig.add_subplot(gs[1, 2*i:2*i+2])

        column_axes['predict_ir'] = fig.add_subplot(gs[2, 2*i])
        column_axes['predict_xuv'] = fig.add_subplot(gs[2, 2*i+1])

        column_axes['reconstruct'] = fig.add_subplot(gs[3, 2*i:2*i+2])

        plot_rows.append(column_axes)

    return plot_rows, fig




def unsupervised_plot():

    global sess
    global trace
    global actual_fields
    global axes
    global i

    i = 0

    # make unsupervised learning plot
    modelname = 'unsupervised_highres'

    # get the trace
    trace, actual_fields = unsupervised.get_trace(index=2, filename='attstrace_test2_processed.hdf5')

    # create plot axes
    axes = create_plots()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './models/{}.ckpt'.format(modelname + '_unsupervised'))

        generate_images_and_plot()




def plot_predictions2(x_in, y_in, pred_in, indexes, axes, figure, epoch, set):

    # get find where in the vector is the ir and xuv


    for j, index in enumerate(indexes):

        prediction = pred_in[index]
        mse = sess.run(network2.loss, feed_dict={network2.x: x_in[index].reshape(1, -1),network2.y_true: y_in[index].reshape(1, -1)})
        # print(mse)
        # print(str(mse))


        xuv_in, ir_in = network2.separate_xuv_ir_vec(y_in[index])
        xuv_pred, ir_pred = network2.separate_xuv_ir_vec(pred_in[index])


        axes[j]['input_trace'].cla()
        axes[j]['input_trace'].pcolormesh(x_in[index].reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values)), cmap='jet')
        axes[j]['input_trace'].text(0.0, 1.1, 'input_trace', transform=axes[j]['input_trace'].transAxes,backgroundcolor='white')
        axes[j]['input_trace'].set_xticks([])
        axes[j]['input_trace'].set_yticks([])



        axes[j]['actual_xuv'].cla()
        xuv_fmat_si = crab_tf2.xuv.f_cropped * sc.physical_constants['atomic unit of time'][0]
        axes[j]['actual_xuv'].plot(xuv_fmat_si, np.abs(xuv_in)**2, color='black')
        axtwin = axes[j]['actual_xuv'].twinx()
        crop_phase = 2
        axtwin.plot(xuv_fmat_si[crop_phase:-crop_phase], np.unwrap(np.angle(xuv_in[crop_phase:-crop_phase])),
                    color='green')
        #axtwin.set_ylim(-2 * np.pi, 2 * np.pi)
        axtwin.set_ylabel('$\phi$')
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        axes[j]['actual_xuv'].text(0.0, 1.1, 'Actual XUV', transform=axes[j]['actual_xuv'].transAxes, backgroundcolor='white')
        axes[j]['actual_xuv'].set_xlabel('Hz')




        axes[j]['predict_xuv'].cla()
        xuv_fmat_si = crab_tf2.xuv.f_cropped * sc.physical_constants['atomic unit of time'][0]
        axes[j]['predict_xuv'].plot(xuv_fmat_si, np.abs(xuv_pred) ** 2, color='black')
        axtwin = axes[j]['predict_xuv'].twinx()
        crop_phase = 2
        axtwin.plot(xuv_fmat_si[crop_phase:-crop_phase], np.unwrap(np.angle(xuv_pred[crop_phase:-crop_phase])),
                    color='green')
        # axtwin.set_ylim(-2 * np.pi, 2 * np.pi)
        axtwin.set_ylabel('$\phi$')
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        axes[j]['predict_xuv'].text(0.0, 1.1, 'Predicted XUV', transform=axes[j]['predict_xuv'].transAxes,
                                   backgroundcolor='white')
        axes[j]['predict_xuv'].set_xlabel('Hz')




        #axes[j]['predict_xuv'].cla()
        #axes[j]['predict_xuv'].plot(np.real(xuv_pred), color='blue')
        #axes[j]['predict_xuv'].plot(np.imag(xuv_pred), color='red')
        #axes[j]['predict_xuv'].text(0.0, 1.1, 'predict_xuv', transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
        #axes[j]['predict_xuv'].set_xticks([])
        #axes[j]['predict_xuv'].set_yticks([])

        axes[j]['actual_ir'].cla()
        ir_fmat_si = crab_tf2.ir.f_cropped / sc.physical_constants['atomic unit of time'][0]
        axes[j]['actual_ir'].plot(ir_fmat_si, np.abs(ir_in)**2, color='black')
        axes[j]['actual_ir'].set_ylim(0, 0.55)
        axtwin =axes[j]['actual_ir'].twinx()
        crop_phase = 2
        axtwin.plot(ir_fmat_si[crop_phase:-crop_phase], np.unwrap(np.angle(ir_in[crop_phase:-crop_phase])), color='green')
        axtwin.set_ylim(-2 * np.pi, 2 * np.pi)
        axtwin.set_ylabel('$\phi$')
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        axes[j]['actual_ir'].text(0.0, 1.1, 'Actual IR', transform=axes[j]['actual_ir'].transAxes, backgroundcolor='white')



        axes[j]['predict_ir'].cla()
        ir_fmat_si = crab_tf2.ir.f_cropped / sc.physical_constants['atomic unit of time'][0]
        axes[j]['predict_ir'].plot(ir_fmat_si, np.abs(ir_in) ** 2, color='black')
        axes[j]['predict_ir'].set_ylim(0, 0.55)
        axtwin = axes[j]['predict_ir'].twinx()
        crop_phase = 2
        axtwin.plot(ir_fmat_si[crop_phase:-crop_phase], np.unwrap(np.angle(ir_in[crop_phase:-crop_phase])),
                    color='green')
        axtwin.set_ylim(-2 * np.pi, 2 * np.pi)
        axtwin.set_ylabel('$\phi$')
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        axes[j]['predict_ir'].text(0.0, 1.1, 'Predicted IR', transform=axes[j]['predict_ir'].transAxes,
                                  backgroundcolor='white')




        #axes[j]['predict_ir'].cla()
        #axes[j]['predict_ir'].plot(ir_fmat_si, np.abs(ir_pred) ** 2, color='black')
        #axes[j]['predict_ir'].text(0.0, 1.1, 'Predicted IR', transform=axes[j]['predict_ir'].transAxes,backgroundcolor='white')
        #axes[j]['predict_ir'].set_xticks([])
        #axes[j]['predict_ir'].set_yticks([])

        # calculate generated streaking trace
        generated_trace = sess.run(crab_tf2.image, feed_dict={crab_tf2.ir_cropped_f: ir_pred,
                                                              crab_tf2.xuv_cropped_f: xuv_pred})

        axes[j]['reconstruct'].pcolormesh(generated_trace,cmap='jet')
        axes[j]['reconstruct'].text(0.0, 1.1, 'reconstructed_trace', transform=axes[j]['reconstruct'].transAxes,backgroundcolor='white')

        axes[j]['reconstruct'].text(0.5, 1.25, 'MSE: {} '.format(str(mse)),
                                    transform=axes[j]['reconstruct'].transAxes, backgroundcolor='white',
                                    ha='center')
        axes[j]['reconstruct'].set_xticks([])
        axes[j]['reconstruct'].set_yticks([])



def supervised_plot():
    i = 0
    global sess

    get_data = network2.GetData(batch_size=10)

    with tf.Session() as sess:

        saver = tf.train.Saver()
        modelname = 'unsupervised_highres'

        # this restores the unsupervised learning trained model
        #saver.restore(sess, './models/{}.ckpt'.format(modelname + '_unsupervised'))

        saver.restore(sess, './models/{}.ckpt'.format(modelname))


        # get data
        batch_x_test, batch_y_test = get_data.evaluate_on_test_data()
        predictions = sess.run(network2.y_pred, feed_dict={network2.x: batch_x_test})

        axes, fig = create_sample_plot()

        plot_predictions2(x_in=batch_x_test, y_in=batch_y_test, pred_in=predictions, indexes=[0, 1, 2],
                          axes=axes, figure=fig, epoch=i + 1, set='train_data_1')

        plt.show()






if __name__ == "__main__":

    unsupervised_plot()

    #tf.reset_default_graph()

    #supervised_plot()




















