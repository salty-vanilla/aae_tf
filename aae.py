import tensorflow as tf
import os
import csv
import time
import numpy as np
from PIL import Image


class AdversarialAutoEncoder:
    def __init__(self, autoencoder, discriminator, is_training=True):
        tf.reset_default_graph()
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.image_shape = self.autoencoder.input_shape
        self.latent_dim = self.autoencoder.latent_dim
        self.image = tf.placeholder(tf.float32, [None] + list(self.image_shape), name='x')
        self.latent_vector = tf.placeholder(tf.float32, [None, self.latent_dim], name='z')

        self.encode, self.decode = self.autoencoder(self.image)

        self.discriminate_real = self.discriminator(self.latent_vector, reuse=False)
        self.discriminate_fake = self.discriminator(self.encode, reuse=True)

        with tf.name_scope('Loss'):
            self.loss_ae = tf.reduce_mean(tf.square(self.image - self.decode), name='MSE')

            self.loss_d_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminate_real,
                                                                       labels=tf.ones_like(self.discriminate_real))
            self.loss_d_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminate_fake,
                                                                       labels=tf.zeros_like(self.discriminate_fake))
            self.loss_d = (tf.reduce_mean(self.loss_d_real) + tf.reduce_mean(self.loss_d_fake)) / 2

            self.loss_en = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminate_fake,
                                                        labels=tf.ones_like(self.discriminate_fake)))

        if is_training:
            with tf.name_scope('Optimizer'):
                self.opt_ae = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).\
                    minimize(self.loss_ae, var_list=self.autoencoder.vars)

                self.opt_d = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5). \
                    minimize(self.loss_d, var_list=self.discriminator.vars)

                self.opt_en = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5). \
                    minimize(self.loss_en, var_list=self.autoencoder.encoder_vars)

        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('../logs', graph=self.sess.graph)
        self.model_dir = None
        self.is_training = is_training

    def fit_generator(self, image_sampler, latent_sampler, nb_epoch=1000, save_steps=5, visualize_steps=5,
                      result_dir='result', model_dir='model'):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample
        self.model_dir = model_dir

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        f = open(os.path.join(result_dir, 'learning_log.csv'), 'w')
        writer = csv.writer(f, lineterminator='\n')

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        writer.writerow(['loss_ae', 'loss_en', 'loss_d'])

        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                image_batch = image_sampler()
                latent_batch = latent_sampler(image_batch.shape[0], self.latent_dim)

                _, loss_ae = self.sess.run([self.opt_ae, self.loss_ae],
                                           feed_dict={self.image: image_batch})

                _, loss_d = self.sess.run([self.opt_d, self.loss_d],
                                          feed_dict={self.image: image_batch,
                                                     self.latent_vector: latent_batch})

                _, loss_en = self.sess.run([self.opt_en, self.loss_en],
                                           feed_dict={self.image: image_batch})

                writer.writerow([loss_ae, loss_en, loss_d])
                
                print('iter : {} / {}  {:.1f}[s]  loss_mse : {:.4f}  loss_d : {:.4f}  loss_en : {:.4f}\r'
                      .format(iter_, steps_per_epoch, time.time() - start, loss_ae, loss_d, loss_en),
                      end='')

            if epoch % visualize_steps == 0:
                self.visualize(os.path.join(result_dir, 'epoch_{}'.format(epoch)),
                               image_batch, image_sampler.data_to_image)

            if epoch % save_steps == 0:
                self.save(epoch)

        print('\nTraining is done ...\n')

    def restore(self, file_path):
        reader = tf.train.NewCheckpointReader(file_path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        var_dict = dict(zip(map(lambda x:
                                x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                current_var = var_dict[saved_var_name]
                var_shape = current_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(current_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, file_path)

    def predict(self, x, batch_size=16):
        outputs = np.empty([0] + list(self.image_shape))
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.predict_on_batch(x_batch)
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def predict_on_batch(self, x):
        return self.sess.run(self.decode,
                             feed_dict={self.image: x})

    def predict_latent_vectors_generator(self, image_sampler):
        outputs = np.empty([0, self.latent_dim])
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        for x_batch in image_sampler():
            o = self.predict_latent_vectors_on_batch(x_batch)
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def predict_latent_vectors(self, x, batch_size=16):
        outputs = np.empty([0, self.latent_dim])
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.predict_latent_vectors_on_batch(x_batch)
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def predict_latent_vectors_on_batch(self, x):
        return self.sess.run(self.encode,
                             feed_dict={self.image: x})

    def visualize(self, dst_dir, image_batch, convert_function):
        decoded_data = self.predict_on_batch(image_batch)
        decoded_images = convert_function(decoded_data)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for i, image in enumerate(decoded_images):
            if image.shape[2] == 1:
                image = image.reshape(image.shape[:2])
            dst_path = os.path.join(dst_dir, "{}.png".format(i))
            pil_image = Image.fromarray(np.uint8(image))
            pil_image.save(dst_path)

    def save(self, epoch):
        dst_dir = os.path.join(self.model_dir, "epoch_{}".format(epoch))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        return self.saver.save(self.sess, save_path=os.path.join(dst_dir, 'model.ckpt'))
