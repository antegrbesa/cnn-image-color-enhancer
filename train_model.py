import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from scipy import misc
import numpy as np
import sys

from load_dataset import load_test_data, load_batch
import models
import utils

# defining size of the training image patches

PATCH_WIDTH = 360
PATCH_HEIGHT = 240
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

batch_size, train_size, learning_rate, num_train_iters, \
w_color, dped_dir, eval_step = utils.process_command_args(sys.argv)

np.random.seed(0)

# loading training and test data

print("Loading test data...")
test_data, test_answ = load_test_data(PATCH_SIZE)
print("Test data was loaded\n")

print("Loading training data...")
train_data, train_answ = load_batch(dped_dir, train_size, PATCH_SIZE)
print("Training data was loaded\n")

TEST_SIZE = test_data.shape[0]
num_test_batches = int(test_data.shape[0] / batch_size)

# defining system architecture

with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
    # placeholders for training data

    bad_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    bad_image = tf.reshape(bad_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    orig_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    orig_image = tf.reshape(orig_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    # get processed enhanced image

    enhanced = models.resnet(bad_image)

    # transform both orig and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    orig_gray = tf.reshape(tf.image.rgb_to_grayscale(orig_image), [-1, PATCH_WIDTH * PATCH_HEIGHT])

    #  color loss

    enhanced_blur = utils.blur(enhanced)
    orig_blur = utils.blur(orig_image)

    loss_color = tf.reduce_sum(tf.pow(orig_blur - enhanced_blur, 2)) / (2 * batch_size)
    loss_generator = w_color * loss_color

    # optimize parameters of image enhancement

    generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
    train_step_gen = tf.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)

    saver = tf.train.Saver(var_list=generator_vars, max_to_keep=100)

    print('Initializing variables')
    sess.run(tf.global_variables_initializer())

    print('Training network')

    train_loss_gen = 0.0

    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])
    test_crops = test_data[np.random.randint(0, TEST_SIZE, 5), :]

    logs = open('models/losses.txt', "w+")
    logs.close()

    for i in range(num_train_iters):

        # train generator

        idx_train = np.random.randint(0, train_size, batch_size)

        bad_images = train_data[idx_train]
        orig_images = train_answ[idx_train]

        [loss_temp, temp] = sess.run([loss_generator, train_step_gen],
                                     feed_dict={bad_: bad_images, orig_: orig_images})
        train_loss_gen += loss_temp / eval_step

        if i % eval_step == 0:

            # test generator

            test_losses_gen = np.zeros((1, 6))

            for j in range(num_test_batches):
                be = j * batch_size
                en = (j + 1) * batch_size

                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                bad_images = test_data[be:en]
                orig_images = test_answ[be:en]

                [enhanced_crops, losses] = sess.run([enhanced, \
                                                                   [loss_generator,loss_color]], \
                                                                feed_dict={bad_: bad_images, orig_: orig_images})

            logs_gen = "generator losses | train: %.4g, test: %.4g |  color: %.4g \n" % \
                       (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][2])
            print(logs_gen)

            # save the results to log file

            logs = open('models/results'+i+'.txt', "a")
            logs.write(logs_gen)
            logs.write('\n')
            logs.close()

            # save visual results for several test image crops

            enhanced_crops = sess.run(enhanced, feed_dict={bad_: test_crops, orig_: orig_images})

            idx = 0
            for crop in enhanced_crops:
                before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                misc.imsave('results/result_ ' + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                idx += 1

            train_loss_gen = 0.0

            # save the model that corresponds to the current iteration

            saver.save(sess, 'models/results_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

            # reload a different batch of training data

            del train_data
            del train_answ
            train_data, train_answ = load_batch(dped_dir, train_size, PATCH_SIZE)
