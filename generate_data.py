from crab_tf import *
import tensorflow as tf
import crab_tf




xuv_test = XUV_Field(N=N, tmax=tmax, gdd=500, tod=0)
xuv_test_input = xuv_test.Et[lower:upper]

init = tf.global_variables_initializer()
with tf.Session() as sess:

    init.run()

    strace = sess.run(image, feed_dict={xuv_input: xuv_test_input.reshape(1, -1, 1)})

    plt.figure(10)
    plt.pcolormesh(tauvec, p_vec, strace, cmap='jet')
    plt.text(0.1, 0.92, 'GDD: {}'.format(xuv_test.gdd_si) + ' $as^2$',
             transform=plt.gca().transAxes, backgroundcolor='white')
    plt.text(0.1, 0.85, 'TOD: {}'.format(xuv_test.tod_si) + ' $as^3$',
             transform=plt.gca().transAxes, backgroundcolor='white')
    plt.savefig('./tracegdd{}tod{}.png'.format(int(xuv_test.gdd_si), int(xuv_test.tod_si)))
    plt.show()




