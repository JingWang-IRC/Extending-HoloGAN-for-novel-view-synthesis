from __future__ import division
import os
import sys
from glob import glob
import json
import shutil
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math

with open(sys.argv[1], 'r') as fh:
    cfg=json.load(fh)
IMAGE_PATH = cfg['image_path']
OUTPUT_DIR = cfg['output_dir']
LOGDIR = os.path.join(OUTPUT_DIR, "log")


from tools.ops import *
from tools.utils import get_image_preprocessed, get_image, merge, inverse_transform, to_bool
from tools.rotation_utils import *
from tools.model_utils import transform_voxel_to_match_image


#----------------------------------------------------------------------------

class HoloGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         output_height=64, output_width=64,
         gf_dim=64, df_dim=64,
         c_dim=3, dataset_name='lsun',
         input_fname_pattern='*.webp'):

    self.sess = sess
    self.crop = crop

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.gf_dim = gf_dim
    self.df_dim = df_dim
    self.c_dim = c_dim

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    #self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
    self.data = glob.glob(IMAGE_PATH + "*")
    self.checkpoint_dir = LOGDIR

  # this just calls build_HoloGAN (a function of this class)
  def build(self, build_func_name):
      build_func = eval("self." + build_func_name)
      build_func()

  # build the main body of HoloGAN. the variable, loss function, network and so on.
  def build_HoloGAN(self):
    self.view_in = tf.placeholder(tf.float32, [None, 6], name='view_in')
    self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')
    self.z = tf.placeholder(tf.float32, [None, cfg['z_dim']], name='z')

    # the next line was aiming to use the idea of intermediate latent space in StyleGAN, but did not work well
    # so, comment out
    #mlp_func = self.mlp_z_to_w

    gen_func = eval("self." + (cfg['generator']))        # generator_AdaIN or generator_AdaIN_res128
    dis_func = eval("self." + (cfg['discriminator']))    # discriminator_IN or discriminator_IN_style_res128


    self.gen_view_func = eval(cfg['view_func'])          # generate_random_rotation_translation

    # HoloGAN did not considered the mapping from Z space to W space (you can refer to StyleGAN for Z and W space)
    # so directly set w = z
    self.w = self.z

    # these z_mapping_functions are affine transformation
    s0, b0 = self.z_mapping_function(self.w, self.gf_dim * 8, 'g_z0')
    s1, b1 = self.z_mapping_function(self.w, self.gf_dim * 4, 'g_z1')
    s2, b2 = self.z_mapping_function(self.w, self.gf_dim * 2, 'g_z2')
    s4, b4 = self.z_mapping_function(self.w, self.gf_dim * 4, 'g_z4')
    s5, b5 = self.z_mapping_function(self.w, self.gf_dim, 'g_z5')
    s6, b6 = self.z_mapping_function(self.w, self.gf_dim // 2, 'g_z6')

    # to update latent vector in W+ space. more details in paper "image2StyleGAN"
    self.w_plus = tf.concat([s0, b0, s1, b1, s2, b2, s4, b4, s5, b5, s6, b6], 1)

    self.G = gen_func(self.w_plus, self.view_in)

    if str.lower(str(cfg["style_disc"])) == "true":      # use style distance
        print("Style Disc")

        # _r: real data   _f: fake data
        # outputs of every layers (h1~h4) will be used for discrimination
        self.D, self.D_logits, self.Q_c, self.d_h1_r, self.d_h2_r, self.d_h3_r, self.d_h4_r, self.d_h3_original, self.d_h4_original = dis_func(self.inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f, self.d_h4_f, _, _ = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)

        self.d_h1_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
        self.d_h2_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
        self.d_h3_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
        self.d_h4_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.ones_like(self.d_h4_r))) \
                    + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_f, tf.zeros_like(self.d_h4_f))))

        # the loss below from style discriminator for generator did not be considered in the source code of HoloGAN
        # I added this and found it worked well
        self.g_d_h1_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.ones_like(self.d_h1_f))))
        self.g_d_h2_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.ones_like(self.d_h2_f))))
        self.g_d_h3_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.ones_like(self.d_h3_f))))
        self.g_d_h4_loss = cfg["DStyle_lambda"] * (
                    tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_f, tf.ones_like(self.d_h4_f))))

    else:
        self.D, self.D_logits, _ = dis_func(self.inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D_, self.D_logits_, self.Q_c_given_x = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)


    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))


    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style disc")
        self.d_loss = self.d_loss + self.d_h1_loss + self.d_h2_loss + self.d_h3_loss + self.d_h4_loss
        self.g_loss = self.g_loss + self.g_d_h1_loss + self.g_d_h2_loss + self.g_d_h3_loss + self.g_d_h4_loss
    #====================================================================================================================
    #Identity loss

    self.q_loss = cfg["lambda_latent"] * tf.reduce_mean(tf.square(self.Q_c_given_x - self.z))
    self.d_loss = self.d_loss + self.q_loss
    self.g_loss = self.g_loss + self.q_loss

    # summary_scalar is tf.summary.scalar or tf.summary_scalar that summaries data for visualisation
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    # specify the trainable variables
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver(max_to_keep=100)

  # train HoloGAN
  def train_HoloGAN(self, config):
      self.d_lr_in = tf.placeholder(tf.float32, None, name='d_eta')
      self.g_lr_in = tf.placeholder(tf.float32, None, name='d_eta')
      #self.data = glob.glob(os.path.join(IMAGE_PATH, "*" + self.input_fname_pattern))
      self.data = glob.glob(IMAGE_PATH + "*")

      # the original code of HoloGAN did not use self.d_lr_in and self.g_lr_in in optimizer
      # I fixed the problem
      d_optim = tf.train.AdamOptimizer(self.d_lr_in, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss, var_list=self.d_vars)
      g_optim = tf.train.AdamOptimizer(self.g_lr_in, beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss, var_list=self.g_vars)

      tf.global_variables_initializer().run()

      # merge_summary is tf.summary_merge     SummaryWriter is tf.train.SummaryWriter. Both summary data for tensorboard
      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
      self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter(LOGDIR, self.sess.graph)

      # Sample noise Z and view parameters to test during training
      sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))    # from uniform distribution. Do not need batch_size because it is default in self.sampling_Z
      sample_view = self.gen_view_func(cfg['batch_size'],               # generate_random_rotation_translation in rotation_utils
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=False,
                                       with_scale=to_bool(str(cfg['with_scale'])))
      sample_files = self.data[0:cfg['batch_size']]
      
      if config.dataset == "cats" or config.dataset == "cars":
          sample_images = [get_image(sample_file,                        # be used in evaluation process
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=False) for sample_file in sample_files]
      else:
          sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=True) for sample_file in sample_files]

      counter = 1
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)

      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      print("Length of data:  ", len(self.data))

      d_lr = cfg['d_eta']
      g_lr = cfg['g_eta']
      for epoch in range(cfg['max_epochs']):

          # decrease the learning rates linearly
          d_lr = d_lr if epoch < cfg['epoch_step'] else d_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])
          g_lr = g_lr if epoch < cfg['epoch_step'] else g_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])

          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size'] : (idx + 1) * cfg['batch_size']]

              # only Compcar dataset was used in my programme, that says "cars" only
              if config.dataset == "cats" or config.dataset == "cars":
                  batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=False) for batch_file in batch_files] # original:crop= False
              else:
                  batch_images = [get_image(batch_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for batch_file in batch_files]
              


              batch_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))    # from uniform distribution
              batch_view = self.gen_view_func(cfg['batch_size'],               # generate_random_rotation_translation in rotation_utils
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=False,
                                       with_scale=to_bool(str(cfg['with_scale'])))

              feed = {self.inputs: batch_images,
                      self.z: batch_z,
                      self.view_in: batch_view,
                      self.d_lr_in: d_lr,
                      self.g_lr_in: g_lr}
              # Update D network
              _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict=feed)
              self.writer.add_summary(summary_str, counter)
              # Update G network
              _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=feed)
              self.writer.add_summary(summary_str, counter)
              # Run g_optim twice
              _, summary_str = self.sess.run([g_optim, self.g_sum],  feed_dict=feed)
              self.writer.add_summary(summary_str, counter)

              errD = self.d_loss.eval(feed)
              errG = self.g_loss.eval(feed)
              errQ = self.q_loss.eval(feed)

              counter += 1
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD, errG, errQ))

              if np.mod(counter, 2000) == 1:
                  self.save(LOGDIR, counter)
                  feed_eval = {self.inputs: sample_images,
                               self.z: sample_z,
                               self.view_in: sample_view,
                               self.d_lr_in: d_lr,
                               self.g_lr_in: g_lr}
                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)
                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                  try:
                      imageio.imwrite(
                          os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)),
                          merge(ren_img, [cfg['batch_size'] // 4, 4]))
                      #scipy.misc.imsave(
                      #    os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)),
                      #    merge(ren_img, [cfg['batch_size'] // 4, 4]))
                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                  except:
                      imageio.imwrite(
                          os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)),
                          ren_img[0])
                      #scipy.misc.imsave(
                      #    os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)),
                      #    ren_img[0])
                      print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

  # the most basic sampling function
  def sample_HoloGAN(self, config):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)
      sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
      if config.rotate_azimuth:
          low  = cfg['azi_low']
          high = cfg['azi_high']
          step = 10
      elif config.rotate_elevation:
          low  = cfg['ele_low']
          high = cfg['ele_high']
          step = 5
      else:
          low  = 0
          high = 10
          step = 1

      for i in range(low, high, step):
          if config.rotate_azimuth:
              sample_view = np.tile(
                  np.array([i * math.pi / 180.0, 20 * math.pi / 180.0, 1.1, 0, 0, 0]), (cfg['batch_size'], 1))
          elif config.rotate_elevation:
              sample_view = np.tile(
                  np.array([180 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.3, 0, 0, 0]), (cfg['batch_size'], 1))
          else:
              sample_view = self.gen_view_func(cfg['batch_size'],
                                               cfg['ele_low'], cfg['ele_high'],
                                               cfg['azi_low'], cfg['azi_high'],
                                               cfg['scale_low'], cfg['scale_high'],
                                               cfg['x_low'], cfg['x_high'],
                                               cfg['y_low'], cfg['y_high'],
                                               cfg['z_low'], cfg['z_high'],
                                               with_translation=False,
                                               with_scale=to_bool(str(cfg['with_scale'])))

          feed_eval = {self.z: sample_z,
                       self.view_in: sample_view}

          samples = self.sess.run(self.G, feed_dict=feed_eval)
          ren_img = inverse_transform(samples)
          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
          try:
              print(os.path.join(SAMPLE_DIR, "{0}_samples_{1:04d}.png".format(counter, i)))
              imageio.imwrite(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1:04d}.png".format(counter, i)),
                  merge(ren_img, [cfg['batch_size'] // 4, 4]))
          except:
              imageio.imwrite(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)),
                  ren_img[0])

  # feed real image to the encoder network to get embedded latent vector
  # use the embedded latent vector as input to generate image
  # generated images gather in a tuple
  def sample_HoloGAN_target_image_tuple(self, config):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)

      # you should set this path to a folder which contains real images (the image you want to change viewpoint)
      images_path = sorted(glob.glob("./sample_test_image/*"))
      target_images = [get_image(image,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=False) for image in images_path[:cfg['batch_size']]]
      feed_eval_z = {self.inputs: target_images}
      sample_z = self.sess.run(self.Q_c, feed_dict=feed_eval_z)
      sample_z = tf.cast(sample_z, tf.float32)
      sample_z_padding = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))[:cfg['batch_size'] - len(target_images), :]
      sample_z = tf.concat((sample_z, sample_z_padding), 0) # align the number to batch size
      sample_z = sample_z.eval(session=self.sess) # change tensor object to numpy (necessary)


      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples_target_tuple")
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)

      if config.rotate_azimuth:
          low  = cfg['azi_low']
          high = cfg['azi_high']
          step = 10
      elif config.rotate_elevation:
          low  = cfg['ele_low']
          high = cfg['ele_high']
          step = 5
      else:
          low  = 0
          high = 10
          step = 1

      print(os.path.join(SAMPLE_DIR, "target.png"))
      imageio.imwrite(
          os.path.join(SAMPLE_DIR, "target.png"),
          merge(np.array(target_images), [cfg['batch_size'] // 4, 4]))

      for i in range(low, high, step):
          if config.rotate_azimuth:
              sample_view = np.tile(
                  np.array([i * math.pi / 180.0, 20 * math.pi / 180.0, 1.1, 0, 0, 0]), (cfg['batch_size'], 1))
          elif config.rotate_elevation:
              sample_view = np.tile(
                  np.array([180 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.3, 0, 0, 0]), (cfg['batch_size'], 1))
          else:
              sample_view = self.gen_view_func(cfg['batch_size'],
                                               cfg['ele_low'], cfg['ele_high'],
                                               cfg['azi_low'], cfg['azi_high'],
                                               cfg['scale_low'], cfg['scale_high'],
                                               cfg['x_low'], cfg['x_high'],
                                               cfg['y_low'], cfg['y_high'],
                                               cfg['z_low'], cfg['z_high'],
                                               with_translation=False,
                                               with_scale=to_bool(str(cfg['with_scale'])))

          feed_eval = {self.z: sample_z,
                       self.view_in: sample_view}
          samples = self.sess.run(self.G, feed_dict=feed_eval)
          ren_img = inverse_transform(samples)
          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)

          try:
              print(os.path.join(SAMPLE_DIR, "{0}_samples_{1:04d}.png".format(counter, i)))
              imageio.imwrite(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1:04d}.png".format(counter, i)),
                  merge(ren_img, [cfg['batch_size'] // 4, 4]))

          except:
              imageio.imwrite(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)),
                  ren_img[0])

  # feed real image to the encoder network to get embedded latent vector
  # use the embedded latent vector as input to generate image
  # generated images are divided into several folders
  def sample_HoloGAN_target_image_divided(self, config):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      # you should set this path to a folder which contains real images (the image you want to change viewpoint)
      images_path = sorted(glob.glob("./sample_test_image/*"))
      target_images = [get_image(image,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=False) for image in images_path[:cfg['batch_size']]]
      feed_eval_z = {self.inputs: target_images}
      sample_z = self.sess.run(self.Q_c, feed_dict=feed_eval_z)
      sample_z = tf.cast(sample_z, tf.float32)
      sample_z = sample_z.eval(session=self.sess) # change tensor object to numpy (necessary)


      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples_target_divided/")
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)

      if config.rotate_azimuth:
          low  = cfg['azi_low']
          high = cfg['azi_high']
          step = 10
      elif config.rotate_elevation:
          low  = cfg['ele_low']
          high = cfg['ele_high']
          step = 5
      else:
          low  = 0
          high = 10
          step = 1

      for i in range(low, high, step):
          if config.rotate_azimuth:
              sample_view = np.tile(
                  np.array([i * math.pi / 180.0, 20 * math.pi / 180.0, 1.1, 0, 0, 0]), (cfg['batch_size'], 1))
          elif config.rotate_elevation:
              sample_view = np.tile(
                  np.array([180 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.3, 0, 0, 0]), (cfg['batch_size'], 1))
          else:
              sample_view = self.gen_view_func(cfg['batch_size'],
                                               cfg['ele_low'], cfg['ele_high'],
                                               cfg['azi_low'], cfg['azi_high'],
                                               cfg['scale_low'], cfg['scale_high'],
                                               cfg['x_low'], cfg['x_high'],
                                               cfg['y_low'], cfg['y_high'],
                                               cfg['z_low'], cfg['z_high'],
                                               with_translation=False,
                                               with_scale=to_bool(str(cfg['with_scale'])))

          feed_eval = {self.z: sample_z,
                       self.view_in: sample_view}

          samples = self.sess.run(self.G, feed_dict=feed_eval)
          ren_img = inverse_transform(samples)

          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
          for idx, img in enumerate(ren_img):
              save_path = SAMPLE_DIR + str(idx)
              if not os.path.exists(save_path):
                  os.mkdir(save_path)
              target_save_path = os.path.join(save_path, "target_{0:04d}.png".format(idx))
              if not os.path.exists(target_save_path):
                  imageio.imwrite(target_save_path, target_images[idx])
              print(os.path.join(save_path, "{0}_samples_{1:04d}_{2:04d}.png".format(counter, i, idx)))
              imageio.imwrite(
                  os.path.join(save_path, "{0}_samples_{1:04d}_{2:04d}.png".format(counter, i, idx)),
                  ren_img[idx])

  # feed real image to the encoder network to get embedded latent vector
  # before feeding the embedded latent vector, the latent vector will be updated in W+ space (paper image2StyleGAN)
  # use the embedded latent vector as input to generate image
  # generated images gather in a tuple
  def sample_HoloGAN_target_image_update_tuple(self, config):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)

      # you should set this path to a folder which contains real images (the image you want to change viewpoint)
      images_path = sorted(glob.glob("./sample_test_image/*"))

      target_images = [get_image(image,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=False) for image in images_path[:cfg['batch_size']]]

      feed_eval_z = {self.inputs: target_images}

      # sample_feature_h3, sample_feature_h4 will be used to measure difference on feature level
      sample_z, sample_feature_h3, sample_feature_h4 = self.sess.run([self.Q_c, self.d_h3_original, self.d_h4_original], feed_dict=feed_eval_z)
      sample_z = tf.cast(sample_z, tf.float32)
      if len(target_images) < cfg['batch_size']:
          sample_z_padding = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))[:cfg['batch_size'] - len(target_images), :]
          sample_z = tf.concat((sample_z, sample_z_padding), 0)
      sample_z = sample_z.eval(session=self.sess) # change tensor object to numpy (necessary)

      input_z = tf.Variable(sample_z, name="z_", dtype=float)
      sample_view = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=False,
                                       with_scale=to_bool(str(cfg['with_scale'])))
      input_view = tf.Variable(sample_view, name="view_update", dtype=float)
      input_images = tf.convert_to_tensor(np.array(target_images), dtype=tf.float32)

      self.sess.run(input_z.initializer)
      self.sess.run(input_view.initializer)

      target_s0, target_b0 = self.z_mapping_function(input_z, self.gf_dim * 8, 'g_z0', reuse=True)
      target_s1, target_b1 = self.z_mapping_function(input_z, self.gf_dim * 4, 'g_z1', reuse=True)
      target_s2, target_b2 = self.z_mapping_function(input_z, self.gf_dim * 2, 'g_z2', reuse=True)
      target_s4, target_b4 = self.z_mapping_function(input_z, self.gf_dim * 4, 'g_z4', reuse=True)
      target_s5, target_b5 = self.z_mapping_function(input_z, self.gf_dim, 'g_z5', reuse=True)
      target_s6, target_b6 = self.z_mapping_function(input_z, self.gf_dim // 2, 'g_z6', reuse=True)

      target_w_plus = tf.concat(
          [target_s0, target_b0, target_s1, target_b1, target_s2, target_b2,
           target_s4, target_b4, target_s5, target_b5, target_s6, target_b6], 1, name="w_plus_update")

      input_w_plus_update = tf.Variable(target_w_plus, name="w_plus_update", dtype=float)
      self.sess.run(input_w_plus_update.initializer)
      generated_images = self.generator_AdaIN_res128(input_w_plus_update, input_view, reuse=True)

      _,_,_,_,_,_,_, gen_feature_h3, gen_feature_h4 = self.discriminator_IN_style_res128(generated_images, cont_dim=cfg['z_dim'], reuse=True)
      sample_feature_h3 = tf.convert_to_tensor(np.array(sample_feature_h3), dtype=tf.float32)
      sample_feature_h4 = tf.convert_to_tensor(np.array(sample_feature_h4), dtype=tf.float32)

      print("type of input_images:  ", type(input_images))
      print("type of generated_images:  ", type(generated_images))


      """ you can use both pix_loss or feature_loss. or only single loss"""
      update_feature_loss = tf.reduce_mean(tf.square(gen_feature_h3 - sample_feature_h3)) + tf.reduce_mean(tf.square(gen_feature_h4 - sample_feature_h4))
      update_pix_loss = tf.reduce_mean(tf.square(generated_images - input_images))
      update_loss = 1.0 * update_pix_loss + 1.0 * update_feature_loss
      update_vars = [var for var in tf.trainable_variables() if '_update' in var.name]

      # the initialisation of adamoptimizer may lead to error, so I used the simplest GradientDescentOptimizer
      #optim = tf.train.AdamOptimizer(0.1, beta1=0.5, beta2=0.999).minimize(vgg_loss, var_list=vgg_vars) # error: beta2 is uninitialized
      optim = tf.train.GradientDescentOptimizer(2).minimize(update_loss, var_list=update_vars)

      time_start = time.time()
      # you can change the iteration number
      for i in range(3):
          print(update_vars)
          print(i)
          pix_loss = update_pix_loss.eval(session=self.sess)
          print("pix_loss:  ", pix_loss )
          feature_loss = update_feature_loss.eval(session=self.sess)
          print("feature_loss:  ", feature_loss)
          self.sess.run(optim)
      time_over = time.time()
      print("time consuming:  ", time_over-time_start)


      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples_target_update_tuple")
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)

      if config.rotate_azimuth:
          low  = cfg['azi_low']
          high = cfg['azi_high']
          step = 10
      elif config.rotate_elevation:
          low  = cfg['ele_low']
          high = cfg['ele_high']
          step = 5
      else:
          low  = 0
          high = 10
          step = 1

      # save the target images as a tuple so that you can compare them visually
      print(os.path.join(SAMPLE_DIR, "target.png"))
      imageio.imwrite(
          os.path.join(SAMPLE_DIR, "target.png"),
          merge(np.array(target_images), [cfg['batch_size'] // 4, 4]))

      for i in range(low, high, step):
          if config.rotate_azimuth:
              sample_view = np.tile(
                  np.array([i * math.pi / 180.0, 20 * math.pi / 180.0, 1.1, 0, 0, 0]), (cfg['batch_size'], 1))
          elif config.rotate_elevation:
              sample_view = np.tile(
                  np.array([180 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.3, 0, 0, 0]), (cfg['batch_size'], 1))
          else:
              sample_view = self.gen_view_func(cfg['batch_size'],
                                               cfg['ele_low'], cfg['ele_high'],
                                               cfg['azi_low'], cfg['azi_high'],
                                               cfg['scale_low'], cfg['scale_high'],
                                               cfg['x_low'], cfg['x_high'],
                                               cfg['y_low'], cfg['y_high'],
                                               cfg['z_low'], cfg['z_high'],
                                               with_translation=False,
                                               with_scale=to_bool(str(cfg['with_scale'])))

          sample_view = tf.convert_to_tensor(np.array(sample_view), dtype=tf.float32)
          samples = self.generator_AdaIN_res128(input_w_plus_update, sample_view, reuse=True)
          samples = self.sess.run(samples)
          ren_img = inverse_transform(samples)
          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
          try:
              print(os.path.join(SAMPLE_DIR, "{0}_samples_{1:04d}.png".format(counter, i)))
              imageio.imwrite(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1:04d}.png".format(counter, i)),
                  merge(ren_img, [cfg['batch_size'] // 4, 4]))
          except:
              imageio.imwrite(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)),
                  ren_img[0])

  # feed real image to the encoder network to get embedded latent vector
  # before feeding the embedded latent vector, the latent vector will be updated in W+ space (paper image2StyleGAN)
  # use the embedded latent vector as input to generate image
  # generated images are divided into several folders
  def sample_HoloGAN_target_image_update_divided(self, config):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)

      # you should set this path to a folder which contains real images (the image you want to change viewpoint)
      images_path = sorted(glob.glob("./sample_test_image/*"))
      target_images = [get_image(image,
                                input_height=self.input_height,
                                input_width=self.input_width,
                                resize_height=self.output_height,
                                resize_width=self.output_width,
                                crop=False) for image in images_path[:cfg['batch_size']]]

      feed_eval_z = {self.inputs: target_images}
      sample_z, sample_feature_h3, sample_feature_h4 = self.sess.run([self.Q_c, self.d_h3_original, self.d_h4_original], feed_dict=feed_eval_z)
      sample_z = tf.cast(sample_z, tf.float32)
      if len(target_images) < cfg['batch_size']:
          sample_z_padding = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))[:cfg['batch_size'] - len(target_images), :]
          sample_z = tf.concat((sample_z, sample_z_padding), 0)
      sample_z = sample_z.eval(session=self.sess) # change tensor object to numpy (necessary)

      input_z = tf.Variable(sample_z, name="z_", dtype=float)
      sample_view = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=False,
                                       with_scale=to_bool(str(cfg['with_scale'])))
      input_view = tf.Variable(sample_view, name="view_update", dtype=float)
      input_images = tf.convert_to_tensor(np.array(target_images), dtype=tf.float32)

      self.sess.run(input_z.initializer)
      self.sess.run(input_view.initializer)

      target_s0, target_b0 = self.z_mapping_function(input_z, self.gf_dim * 8, 'g_z0', reuse=True)
      target_s1, target_b1 = self.z_mapping_function(input_z, self.gf_dim * 4, 'g_z1', reuse=True)
      target_s2, target_b2 = self.z_mapping_function(input_z, self.gf_dim * 2, 'g_z2', reuse=True)
      target_s4, target_b4 = self.z_mapping_function(input_z, self.gf_dim * 4, 'g_z4', reuse=True)
      target_s5, target_b5 = self.z_mapping_function(input_z, self.gf_dim, 'g_z5', reuse=True)
      target_s6, target_b6 = self.z_mapping_function(input_z, self.gf_dim // 2, 'g_z6', reuse=True)

      target_w_plus = tf.concat(
          [target_s0, target_b0, target_s1, target_b1, target_s2, target_b2,
           target_s4, target_b4, target_s5, target_b5, target_s6, target_b6], 1, name="w_plus_update")

      input_w_plus_update = tf.Variable(target_w_plus, name="w_plus_update", dtype=float)
      self.sess.run(input_w_plus_update.initializer)
      generated_images = self.generator_AdaIN_res128(input_w_plus_update, input_view, reuse=True)

      _,_,_,_,_,_,_, gen_feature_h3, gen_feature_h4 = self.discriminator_IN_style_res128(generated_images, cont_dim=cfg['z_dim'], reuse=True)
      sample_feature_h3 = tf.convert_to_tensor(np.array(sample_feature_h3), dtype=tf.float32)
      sample_feature_h4 = tf.convert_to_tensor(np.array(sample_feature_h4), dtype=tf.float32)

      print("type of input_images:  ", type(input_images))
      print("type of generated_images:  ", type(generated_images))


      """ you can both use pix_loss or feature_loss. or only single loss"""
      update_feature_loss = tf.reduce_mean(tf.square(gen_feature_h3 - sample_feature_h3)) + tf.reduce_mean(tf.square(gen_feature_h4 - sample_feature_h4))
      update_pix_loss = tf.reduce_mean(tf.square(generated_images - input_images))
      update_loss = 1.0 * update_pix_loss + 1.0 * update_feature_loss
      update_vars = [var for var in tf.trainable_variables() if '_update' in var.name]

      # the initialisation of adamoptimizer may lead to error, so I used the simplest GradientDescentOptimizer
      #optim = tf.train.AdamOptimizer(0.1, beta1=0.5, beta2=0.999).minimize(vgg_loss, var_list=vgg_vars) # error: beta2 is uninitialized
      optim = tf.train.GradientDescentOptimizer(2).minimize(update_loss, var_list=update_vars)

      time_start = time.time()
      for i in range(3):
          print(update_vars)
          print(i)
          pix_loss = update_pix_loss.eval(session=self.sess)
          print("pix_loss:  ", pix_loss )
          feature_loss = update_feature_loss.eval(session=self.sess)
          print("feature_loss:  ", feature_loss)
          self.sess.run(optim)
      time_over = time.time()
      print("time consuming:  ", time_over-time_start)

      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples_target_update_divided/")
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)

      if config.rotate_azimuth:
          low  = cfg['azi_low']
          high = cfg['azi_high']
          step = 10
      elif config.rotate_elevation:
          low  = cfg['ele_low']
          high = cfg['ele_high']
          step = 5
      else:
          low  = 0
          high = 10
          step = 1

      for i in range(low, high, step):
          if config.rotate_azimuth:
              sample_view = np.tile(
                  np.array([i * math.pi / 180.0, 20 * math.pi / 180.0, 1.1, 0, 0, 0]), (cfg['batch_size'], 1))
          elif config.rotate_elevation:
              sample_view = np.tile(
                  np.array([180 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.3, 0, 0, 0]), (cfg['batch_size'], 1))
          else:
              sample_view = self.gen_view_func(cfg['batch_size'],
                                               cfg['ele_low'], cfg['ele_high'],
                                               cfg['azi_low'], cfg['azi_high'],
                                               cfg['scale_low'], cfg['scale_high'],
                                               cfg['x_low'], cfg['x_high'],
                                               cfg['y_low'], cfg['y_high'],
                                               cfg['z_low'], cfg['z_high'],
                                               with_translation=False,
                                               with_scale=to_bool(str(cfg['with_scale'])))

          sample_view = tf.convert_to_tensor(np.array(sample_view), dtype=tf.float32)
          samples = self.generator_AdaIN_res128(input_w_plus_update, sample_view, reuse=True)
          samples = self.sess.run(samples)
          ren_img = inverse_transform(samples)
          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
          for idx, img in enumerate(ren_img):
              save_path = SAMPLE_DIR + str(idx)
              if not os.path.exists(save_path):
                  os.mkdir(save_path)
              target_save_path = os.path.join(save_path, "target_{0:04d}.png".format(idx))
              if not os.path.exists(target_save_path):
                  imageio.imwrite(target_save_path, target_images[idx])
              print(os.path.join(save_path, "{0}_samples_{1:04d}_{2:04d}.png".format(counter, i, idx)))
              imageio.imwrite(
                  os.path.join(save_path, "{0}_samples_{1:04d}_{2:04d}.png".format(counter, i, idx)),
                  ren_img[idx])

  # sample many images by feeding random latent vectors for computing FID score
  def sample_HoloGAN_many(self, config, amount):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples_many")
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)

      batch_idxs = amount // cfg['batch_size'] + 1

      for idx in range(0, batch_idxs):
          sample_view = self.gen_view_func(cfg['batch_size'],
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=False,
                                       with_scale=to_bool(str(cfg['with_scale'])))
          sample_z = np.random.uniform(-1., 1., (cfg['batch_size'], cfg['z_dim']))
          feed_eval = {self.z: sample_z,
                       self.view_in: sample_view}

          samples = self.sess.run(self.G, feed_dict=feed_eval)
          ren_img = inverse_transform(samples)
          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)

          for i, image in enumerate(ren_img):
              if int(cfg['batch_size'])*idx + i >= amount:
                  break
              try:
                  print(os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(int(cfg['batch_size'])*idx + i)))
                  imageio.imwrite(
                      os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(int(cfg['batch_size'])*idx + i)),
                      image)
              except:
                  imageio.imwrite(
                      os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(int(cfg['batch_size']) * idx + i)),
                      image)

  # sample many images by feeding embedded latent vectors for computing FID score
  def sample_HoloGAN_many_target_image(self, config, amount):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      # you should set this path to a folder which contains real images
      images_path = glob.glob("/home/wang-jing/tensorflow_v1/image_resizer/seperate/test/*")
      #random.shuffle(images_path)
      print(len(images_path))

      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples_many_target_image")
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)

      batch_idxs = amount // cfg['batch_size'] + 1

      for idx in range(0, batch_idxs):
          target_images = [get_image(image,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=False) for image in images_path[idx * cfg['batch_size']: (idx + 1) * cfg['batch_size']]]

          feed_eval_z = {self.inputs: target_images}
          sample_z = self.sess.run(self.Q_c, feed_dict=feed_eval_z)
          sample_z = tf.cast(sample_z, tf.float32)
          sample_z_padding = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))[
                             :cfg['batch_size'] - len(target_images), :]
          sample_z = tf.concat((sample_z, sample_z_padding), 0)
          sample_z = sample_z.eval(session=self.sess)  # change tensor object to numpy (necessary)

          sample_view = self.gen_view_func(cfg['batch_size'],
                                           cfg['ele_low'], cfg['ele_high'],
                                           cfg['azi_low'], cfg['azi_high'],
                                           cfg['scale_low'], cfg['scale_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=False,
                                           with_scale=to_bool(str(cfg['with_scale'])))
          feed_eval = {self.z: sample_z,
                       self.view_in: sample_view}
          samples = self.sess.run(self.G, feed_dict=feed_eval)
          ren_img = inverse_transform(samples)
          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)

          for i, image in enumerate(ren_img):
              if int(cfg['batch_size']) * idx + i >= amount:
                  break
              try:
                  print(os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(int(cfg['batch_size']) * idx + i)))
                  imageio.imwrite(
                      os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(int(cfg['batch_size']) * idx + i)),
                      image)
              except:
                  imageio.imwrite(
                      os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(int(cfg['batch_size']) * idx + i)),
                      image)

  # sample many images by feeding updated embedded latent vectors for computing FID score
  # however, because my poor programming skills, this function may encounter OOM problem
  # the memory will be fulled gradually during runtime
  # so you can try to sample 2,000 images one time and repeat until sampled enough images (it depends on your memory)
  def sample_HoloGAN_many_target_image_update(self, config, start_idx, over_idx, update_num):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      # you should set this path to a folder which contains real images
      images_path = sorted(glob.glob("/home/wang-jing/tensorflow_v1/image_resizer/seperate/test/*"))
      images_path = images_path[start_idx: over_idx]
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return

      save_path = "samples_many_target_image_update" + str(update_num) + "/"
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, save_path)
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)

      amount = over_idx - start_idx
      batch_idxs = amount // cfg['batch_size'] + 1

      for idx in range(0, batch_idxs):
          target_images = [get_image(image,
                             input_height=self.input_height,
                             input_width=self.input_width,
                             resize_height=self.output_height,
                             resize_width=self.output_width,
                             crop=False) for image in
                             images_path[idx * cfg['batch_size']: (idx + 1) * cfg['batch_size']]]

          feed_eval_z = {self.inputs: target_images}
          sample_z, sample_feature_h3, sample_feature_h4 = self.sess.run([self.Q_c, self.d_h3_original, self.d_h4_original], feed_dict=feed_eval_z)
          sample_z = tf.convert_to_tensor(np.array(sample_z), dtype=tf.float32)
          number_input = sample_z.shape[0]
          print("number is:  ", number_input)
          if len(target_images) < cfg['batch_size']:
              sample_z_padding = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))[
                                 :cfg['batch_size'] - len(target_images), :]
              sample_z = tf.concat((sample_z, sample_z_padding), 0)


          target_s0, target_b0 = self.z_mapping_function(sample_z, self.gf_dim * 8, 'g_z0', reuse=True)
          target_s1, target_b1 = self.z_mapping_function(sample_z, self.gf_dim * 4, 'g_z1', reuse=True)
          target_s2, target_b2 = self.z_mapping_function(sample_z, self.gf_dim * 2, 'g_z2', reuse=True)
          target_s4, target_b4 = self.z_mapping_function(sample_z, self.gf_dim * 4, 'g_z4', reuse=True)
          target_s5, target_b5 = self.z_mapping_function(sample_z, self.gf_dim, 'g_z5', reuse=True)
          target_s6, target_b6 = self.z_mapping_function(sample_z, self.gf_dim // 2, 'g_z6', reuse=True)

          target_w_plus = tf.concat(
              [target_s0, target_b0, target_s1, target_b1, target_s2, target_b2,
               target_s4, target_b4, target_s5, target_b5, target_s6, target_b6], 1)

          sample_z = sample_z.eval(session=self.sess)  # change tensor object to numpy (necessary)
          sample_view = self.gen_view_func(cfg['batch_size'],
                                           cfg['ele_low'], cfg['ele_high'],
                                           cfg['azi_low'], cfg['azi_high'],
                                           cfg['scale_low'], cfg['scale_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=False,
                                           with_scale=to_bool(str(cfg['with_scale'])))

          # only set the tf.Variable in the beginning
          # otherwise, the memory will be fulled tremendously fast
          if idx == 0:
              input_z = tf.Variable(sample_z, name="z_", dtype=float)
              input_w_plus_update = tf.Variable(target_w_plus, name="w_plus_update", dtype=float)
              input_view = tf.Variable(sample_view, name="view_update", dtype=float)
              self.sess.run(input_z.initializer)
              self.sess.run(input_w_plus_update.initializer)
              self.sess.run(input_view.initializer)

          input_images = tf.convert_to_tensor(np.array(target_images), dtype=tf.float32)

          update1 = tf.assign(input_z, sample_z)
          update2 = tf.assign(input_w_plus_update, target_w_plus)
          update3 = tf.assign(input_view, sample_view)
          self.sess.run(update1)
          self.sess.run(update2)
          self.sess.run(update3)

          generated_images = self.generator_AdaIN_res128(input_w_plus_update, input_view, reuse=True)
          _, _, _, _, _, _, _, gen_feature_h3, gen_feature_h4 = self.discriminator_IN_style_res128(generated_images, cont_dim=cfg['z_dim'], reuse=True)
          sample_feature_h3 = tf.convert_to_tensor(np.array(sample_feature_h3), dtype=tf.float32)
          sample_feature_h4 = tf.convert_to_tensor(np.array(sample_feature_h4), dtype=tf.float32)


          update_feature_loss = tf.reduce_mean(tf.square(gen_feature_h3[:number_input] - sample_feature_h3[:number_input])) + tf.reduce_mean(
              tf.square(gen_feature_h4[:number_input] - sample_feature_h4[:number_input]))
          update_pix_loss = tf.reduce_mean(tf.square(generated_images[:number_input] - input_images[:number_input]))
          update_loss = 1.0 * update_pix_loss + 1.0 * update_feature_loss
          update_vars = [var for var in tf.trainable_variables() if '_update' in var.name]
          print("this is vgg_vars:  \n", update_vars)

          # only set the optimizer in the beginning
          # otherwise, the memory will be fulled tremendously fast
          if idx == 0:
              optim = tf.train.GradientDescentOptimizer(1).minimize(update_loss, var_list=update_vars)

          for i in range(update_num):
              print(update_vars)
              pix_loss = update_pix_loss.eval(session=self.sess)
              feature_loss = update_feature_loss.eval(session=self.sess)
              print("index: ", idx, "  iteration: ", i, "  feature_loss:  ", feature_loss, "  pix_loss:  ", pix_loss)
              self.sess.run(optim)

          sample_view = self.gen_view_func(cfg['batch_size'],
                                           cfg['ele_low'], cfg['ele_high'],
                                           cfg['azi_low'], cfg['azi_high'],
                                           cfg['scale_low'], cfg['scale_high'],
                                           cfg['x_low'], cfg['x_high'],
                                           cfg['y_low'], cfg['y_high'],
                                           cfg['z_low'], cfg['z_high'],
                                           with_translation=False,
                                           with_scale=to_bool(str(cfg['with_scale'])))

          sample_view = tf.convert_to_tensor(np.array(sample_view), dtype=tf.float32)
          samples = self.generator_AdaIN_res128(input_w_plus_update, sample_view, reuse=True)
          samples = self.sess.run(samples)
          ren_img = inverse_transform(samples)
          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)

          for i, image in enumerate(ren_img):
              if (start_idx + int(cfg['batch_size']) * idx + i) >= over_idx:
                  break
              try:
                  print(os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(start_idx + int(cfg['batch_size']) * idx + i)))
                  imageio.imwrite(
                      os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(start_idx + int(cfg['batch_size']) * idx + i)),
                      image)
              except:
                  imageio.imwrite(
                      os.path.join(SAMPLE_DIR, "samples_{0:04d}.png".format(start_idx + int(cfg['batch_size']) * idx + i)),
                      image)

#=======================================================================================================================

  def sampling_Z(self, z_dim, type="uniform"):
      if str.lower(type) == "uniform":
          return np.random.uniform(-1., 1., (cfg['batch_size'], z_dim))
      else:
          return np.random.normal(0, 1, (cfg['batch_size'], z_dim))

  def linear_classifier(self, features, scope = "lin_class", stddev=0.02, reuse=False):
      with tf.variable_scope(scope) as sc:
          w = tf.get_variable('w', [features.get_shape()[-1], 1],
                              initializer=tf.random_normal_initializer(stddev=stddev))
          b = tf.get_variable('biases', 1, initializer=tf.constant_initializer(0.0))
          logits = tf.matmul(features, w) + b
          return   tf.nn.sigmoid(logits), logits

  def z_mapping_function(self, z, output_channel, scope='z_mapping', act="relu", stddev=0.02, reuse=False):
      with tf.variable_scope(scope) as sc:
          if reuse:
              sc.reuse_variables()
          w = tf.get_variable('w', [z.get_shape()[-1], output_channel * 2],
                              initializer=tf.random_normal_initializer(stddev=stddev))
          b = tf.get_variable('biases', output_channel * 2, initializer=tf.constant_initializer(0.0))
          if act == "relu":
              out = tf.nn.relu(tf.matmul(z, w) + b)
          else:
              out = lrelu(tf.matmul(z, w) + b)
          return out[:, :output_channel], out[:, output_channel:]

#=======================================================================================================================
  # this function did not be used in this programme
  # I tried to imitate StyleGAN to make an internal latent space
  # however, the training did not go well
  def mlp_z_to_w(self, z, z_dim, reuse=False):
      with tf.variable_scope("mlp") as scope:
          if reuse:
              scope.reuse_variables()
          w1 = lrelu((linear(z, z_dim, 'g_mlp_latent_1')))
          w2 = lrelu((linear(w1, z_dim, 'g_mlp_latent_2')))
          w3 = lrelu((linear(w2, z_dim, 'g_mlp_latent_3')))
          w4 = lrelu((linear(w3, z_dim, 'g_mlp_latent_4')))
          w5 = lrelu((linear(w4, z_dim, 'g_mlp_latent_5')))
          return w5

  # this is a discriminator for HoloGAN of 64x64 resolution images
  def discriminator_IN(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
          h1 = lrelu(instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
          h2 = lrelu(instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
          h3 = lrelu(instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))

          #Returning logits to determine whether the images are real or fake
          h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h3), 120, 'd_latent')))
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

  # this is a discriminator for HoloGAN of 128x128 resolution images
  # I used this in my research
  def discriminator_IN_style_res128(self, image,  cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = instance_norm(h1, 'd_in1', True)
          h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          d_h1_style = tf.concat([h1_mean, h1_var], 0)
          d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = instance_norm(h2, 'd_in2', True)
          h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          d_h2_style = tf.concat([h2_mean, h2_var], 0)
          d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = instance_norm(h3, 'd_in3', True)
          h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          d_h3_style = tf.concat([h3_mean, h3_var], 0)
          d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = instance_norm(h4, 'd_in4', True)
          h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          d_h4_style = tf.concat([h4_mean, h4_var], 0)
          d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h4), 120, 'd_latent')))
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits, h3, h4

  # this is a generator for HoloGAN of 64x64 resolution images
  def generator_AdaIN(self, z, view_in, reuse=False):
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3, name='g_h1')
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3, name='g_h2')
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          #=============================================================================================================
          h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)
          #=============================================================================================================
          # Collapsing depth dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = tf.nn.relu(h3)
          #=============================================================================================================

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          return output

  # this is a generator for HoloGAN of 128x128 resolution images
  # I used this in my research
  def generator_AdaIN_res128(self, w_plus, view_in, reuse=False):


      """
      w:         [4, 4, 4, 64*8]
      h0:        [batch_size, 4, 4, 4, 64*8]
      h1:        [batch_size, 8, 8, 8, 64*4]
      h2:        [batch_size, 16, 16, 16, 64*2]
      h2_proj1:  [batch_size, 16, 16, 16, 64]
      h2_proj2:  [batch_size, 16, 16, 16, 64]
      h2_2d:     [batch_size, 16, 16, 16*64]
      h3:        [batch_size, 16, 16, 16*64/2]
      h4:        [batch_size, 32, 32, 64*4]
      h5:        [batch_size, 64, 64, 64]
      h6:        [batch_size, 128, 128, 32]
      h7:        [batch_size, 128, 128, 3]
      """

      #print("==================================================================================")
      #print("check w_plus shape:  ", w_plus.shape)
      tf.print("check w_plus shape:  ", w_plus.shape)

      counter = 0
      s0, b0 = w_plus[:, counter: self.gf_dim * 8], w_plus[:, (counter + self.gf_dim * 8): (counter + self.gf_dim * 8 * 2)]
      counter = counter + self.gf_dim * 8 * 2
      #print("check_after_s0:   ", counter)
      s1, b1 = w_plus[:, counter: (counter + self.gf_dim * 4)], w_plus[:, (counter + self.gf_dim * 4): (counter + self.gf_dim * 4 * 2)]
      counter = counter + self.gf_dim * 4 * 2
      #print("check_after_s1:   ", counter)
      s2, b2 = w_plus[:, counter: (counter + self.gf_dim * 2)], w_plus[:, (counter + self.gf_dim * 2): (counter + self.gf_dim * 2 * 2)]
      counter = counter + self.gf_dim * 2 * 2
      #print("check_after_s2:   ", counter)
      s4, b4 = w_plus[:, counter: (counter + self.gf_dim * 4)], w_plus[:, (counter + self.gf_dim * 4): (counter + self.gf_dim * 4 * 2)]
      counter = counter + self.gf_dim * 4 * 2
      s5, b5 = w_plus[:, counter: (counter + self.gf_dim)], w_plus[:, (counter + self.gf_dim): (counter + self.gf_dim * 2)]
      counter = counter + self.gf_dim * 2
      s6, b6 = w_plus[:, counter: (counter + self.gf_dim // 2) ], w_plus[:, (counter + self.gf_dim // 2): (counter + self.gf_dim // 2 * 2)]

      #print("==================================================================================")
      #print("check s0, b0 shape:  ", s0.shape, "  ", b0.shape)
      #print("==================================================================================")
      #print("check s1, b1 shape:  ", s1.shape, "  ", b1.shape)
      #print("==================================================================================")
      #print("check s2, b2 shape:  ", s2.shape, "  ", b2.shape)
      #print("==================================================================================")
      #print("check s4, b4 shape:  ", s4.shape, "  ", b4.shape)
      #print("==================================================================================")
      #print("check s5, b5 shape:  ", s5.shape, "  ", b5.shape)
      #print("==================================================================================")
      #print("check s6, b6 shape:  ", s6.shape, "  ", b6.shape)

      batch_size = tf.shape(w_plus)[0]
      s_h, s_w, s_d = 64, 64, 64
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)        # 32
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)     # 16
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)     # 8
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)  # 4

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):             # self.gf_dim = 64
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              h0 = AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')
          h1 = AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3, name='g_h2')
          h2 = AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          #=============================================================================================================
          h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')
          h2_proj1 = lrelu( h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')
          h2_proj2 = lrelu( h2_proj2)
          # =============================================================================================================
          # Collapsing depth dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = lrelu(h3)
          # =============================================================================================================

          h4 = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          h4 = AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          h5 = AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          h6 = AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          #print("================================================================================")
          #print("output of gen_res128:  ", output.get_shape().as_list())
          return output

#=======================================================================================================================
  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.dataset_name,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "HoloGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0


