import crab_tf2
import shutil
import network2
import tensorflow as tf
import glob
import tables
import numpy as np
import  matplotlib.pyplot as plt


def get_trace(index, filename):

    with tables.open_file(filename, mode='r') as hdf5_file:

        trace = hdf5_file.root.trace[index,:]

    return trace




# copy the model to a new version to use for unsupervised learning
modelname = '2_test'
for file in glob.glob(r'./models/{}.ckpt.*'.format(modelname)):
    file_newname = file.replace(modelname, modelname+'_unsupervised')
    shutil.copy(file, file_newname)


# get the trace
trace = get_trace(index=0, filename='attstrace_test2_processed.hdf5').reshape(1,-1)



with tf.Session() as sess:

    saver = tf.train.Saver()
    saver.restore(sess, './models/{}.ckpt'.format(modelname+'_unsupervised'))

    # retrieve output from network
    ir_xuv_output = sess.run(network2.y_pred, feed_dict={network2.x: trace})

    # plot streaking trace of the output
    xuv_pred, ir_pred = network2.separate_xuv_ir_vec(ir_xuv_output[0])
    generated_trace = sess.run(crab_tf2.image, feed_dict={crab_tf2.ir_cropped_f: ir_pred,
                                                          crab_tf2.xuv_cropped_f: xuv_pred})








