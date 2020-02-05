from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import csv
import skimage.measure
from data_reader import *
from data_reader_cifar10 import *
from net import *
from utils import *

flags = tf.app.flags
conf = flags.FLAGS


class Solver(object):
    def __init__(self):
        # Initialize  results saving path: model_dir, validate_dir, test_dir
        if not os.path.exists(conf.model_dir):
            os.makedirs(conf.model_dir)
        if not os.path.exists(conf.validate_dir):
            os.makedirs(conf.validate_dir)
        if not os.path.exists(conf.test_dir):
            os.makedirs(conf.test_dir)

    def train(self):
        # Create Session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Initialize gpu device name
        if conf.use_gpu:
            device_str = '/gpu:' + str(conf.device_id)
        else:
            device_str = '/cpu:0'

        with tf.device(device_str):
            # Initialize data loader
            if conf.train_only:
                if conf.dataset == 'cifar10':
                    cifar10 = Cifar10()
                    self.train_hr, self.train_lr, self.val_hr, self.val_lr = cifar10.load_dataset(sess)
                elif conf.dataset == 'DIV2K' or 'MNIST':
                    div2k = DataSet()
                    self.train_hr_file_list, self.train_lr_file_list, self.valid_hr_file_list, self.valid_lr_file_list = div2k.load_file_list(
                        conf.train_hr_list, conf.train_lr_list, conf.val_hr_list, conf.val_lr_list)
                    self.train_hr, self.train_lr, self.val_hr, self.val_lr = div2k.load_dataset(self.train_hr_file_list,
                                                                                                self.train_lr_file_list,
                                                                                                self.valid_hr_file_list,
                                                                                                self.valid_lr_file_list,
                                                                                                sess)
                else:
                    print("No dataset specified.")
                    return

                # Train
                self.img_hr = tf.placeholder(tf.float32)
                self.img_lr = tf.placeholder(tf.float32)
                dataset = tf.data.Dataset.from_tensor_slices((self.img_hr, self.img_lr))
                dataset = dataset.repeat(conf.num_epoch).shuffle(buffer_size=conf.shuffle_size).batch(conf.batch_size)
                self.iterator = dataset.make_initializable_iterator()
                self.next_batch = self.iterator.get_next()

                # Val
                self.val_img_hr = tf.placeholder(tf.float32)
                self.val_img_lr = tf.placeholder(tf.float32)
                val_dataset = tf.data.Dataset.from_tensor_slices((self.val_img_hr, self.val_img_lr))
                val_dataset = val_dataset.repeat().batch(conf.batch_size)
                self.val_iterator = val_dataset.make_initializable_iterator()
                self.val_next_batch = self.val_iterator.get_next()

                # Initialize data stream
                sess.run(self.iterator.initializer, feed_dict={self.img_hr: self.train_hr, self.img_lr: self.train_lr})
                sess.run(self.val_iterator.initializer, feed_dict={self.val_img_hr: self.val_hr, self.val_img_lr: self.val_lr})
            if conf.test_only:
                pass

            # Initialize network
            self.labels = tf.placeholder(tf.float32, shape=[conf.batch_size, conf.hr_size, conf.hr_size, conf.img_channel])
            self.inputs = tf.placeholder(tf.float32, shape=[conf.batch_size, conf.lr_size, conf.lr_size, conf.img_channel])
            self.net = Net(self.labels, self.inputs, mask_type=conf.mask_type,
                           is_linear_only=conf.linear_mapping_only, scope='sr_spc')

        print("----- Done Initialization -----")

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(conf.model_dir, sess.graph)

        learning_rate = tf.train.exponential_decay(conf.learning_rate,
                                                   self.global_step,
                                                   decay_steps=conf.decay_steps,
                                                   decay_rate=conf.decay_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        variables_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sr_spc')
        saver = tf.train.Saver(var_list=variables_list)

        restorer = tf.train.Saver()
        if conf.linear_mapping_only:
            if conf.restore_linear_part:
                linear_mapping_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                             scope='sr_spc/linear_mapping')
                linear_restorer = tf.train.Saver(var_list=linear_mapping_variables)
                linear_restorer.restore(sess, conf.linear_model)
            train_op = optimizer.minimize(self.net.loss, global_step=self.global_step)
        elif conf.train_residual_only:
            linear_mapping_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sr_spc/linear_mapping')
            linear_restorer = tf.train.Saver(var_list=linear_mapping_variables)
            linear_restorer.restore(sess, conf.linear_model)
            first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'sr_spc/residual_blocks')
            train_op = optimizer.minimize(self.net.loss, global_step=self.global_step, var_list=first_train_vars)
        if conf.restore_whole_model:
            restorer.restore(sess, conf.whole_model)
            train_op = optimizer.minimize(self.net.loss, global_step=self.global_step)
        else:
            train_op = optimizer.minimize(self.net.loss, global_step=self.global_step)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # saver.restore(sess, conf.whole_model)

        print("----- Hyperparameter initialized. Begin training -----")

        # Start training
        iters = 1
        try:
            while True:
                hr_img, lr_img = sess.run(self.next_batch)
                # Calculate and optimize loss
                if iters % 100 == 0:
                    t1 = time.time()
                    _, loss, summary_str = sess.run([train_op, self.net.loss, summary_op],
                                                    feed_dict={self.labels: hr_img, self.inputs: lr_img})
                    t2 = time.time()
                    summary_writer.add_summary(summary_str, iters)
                    print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' %
                          (iters, loss, conf.batch_size / (t2 - t1), (t2 - t1)))
                else:
                    t1 = time.time()
                    _, loss = sess.run([train_op, self.net.loss], feed_dict={self.labels: hr_img, self.inputs: lr_img})
                    t2 = time.time()
                    print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' %
                          (iters, loss, conf.batch_size / (t2 - t1), (t2 - t1)))
                iters += 1

                # Run validation
                if iters % 100 == 0:
                    val_hr_img, val_lr_img = sess.run(self.val_next_batch)
                    self.validate(sess, val_hr_img, val_lr_img, summary_writer, iters)

                # Save module
                if iters % 100 == 0:
                    checkpoint_path = os.path.join(conf.model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=iters)
        except tf.errors.OutOfRangeError:
            print('Epoch limit reached')

        # Close session
        sess.close()
        print('----- Done Training -----')

    def validate(self, sess, hr_img, lr_img, summary_writer, step):
        if conf.linear_mapping_only:
            linear_mapping_logits = self.net.linear_mapping_logits
            img_output = tf.clip_by_value(linear_mapping_logits, -1.0, 1.0)
        else:
            residual_reducing_logits = self.net.net_image1.outputs
            img_output = tf.clip_by_value(residual_reducing_logits, -1.0, 1.0)

        # if conf.charbonnier_loss:
        #     loss = self.net.compute_charbonnier_loss(hr_img, img_output, is_mean=True)
        # else:
        #     loss = tf.reduce_mean(tf.nn.l2_loss(tf.losses.absolute_difference(hr_img, img_output)))
        # recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(img_output - hr_img), axis=1))

        recon_loss = self.net.compute_charbonnier_loss(hr_img, img_output, is_mean=True)

        t1 = time.time()
        output, loss_val = sess.run([img_output, recon_loss], feed_dict={self.labels: hr_img, self.inputs: lr_img})
        t2 = time.time()

        process_time = conf.batch_size / (t2 - t1)

        valid_summary = tf.Summary()
        if conf.linear_mapping_only:
            valid_summary.value.add(tag='linear_loss_val', simple_value=loss_val)
        else:
            valid_summary.value.add(tag='residual_loss_val', simple_value=loss_val)
        summary_writer.add_summary(valid_summary, step)

        # Save validated images
        tl.vis.save_images(hr_img, [int(np.ceil(sqrt(conf.batch_size))), int(np.ceil(sqrt(conf.batch_size)))],
                           conf.validate_dir + '/hr_' + str(step) + '.png')
        tl.vis.save_images(output, [int(np.ceil(sqrt(conf.batch_size))), int(np.ceil(sqrt(conf.batch_size)))],
                           conf.validate_dir + '/generate_' + str(step) + '.png')

        # Measure PSNR
        img_psnr = np.zeros(12)
        for i in range(11):
            img_psnr[i] = skimage.measure.compare_psnr(hr_img[i], output[i])

        img_psnr[-1] = np.mean(img_psnr[:11])

        # Save processing time, PSNR
        write_file_obj = open('metrics.csv', 'a')
        writer = csv.writer(write_file_obj)
        writer.writerow([step, loss_val, process_time, img_psnr])
        write_file_obj.close()

        print('----- Done Validation -----')

    def test(self):
        pass
