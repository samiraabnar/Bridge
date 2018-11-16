from mapping_models.basic_mapper import BasicMapper
from tf_layers.linear_mapper import WeightedInputLinearLayer
import tensorflow as tf
import numpy as np

class TfMapper(BasicMapper):
  def __init__(self, hparams, sess):
    self.hparams = hparams
    self.layers = [WeightedInputLinearLayer(scope='linear_layer_1',input_dim=2, output_dim=1, input_len=1)]
    self.build_flag = False
    self.sess = sess

  def build(self, is_train=True):

    self.inputs_var = tf.placeholder(name='input', dtype=tf.float32)
    self.targets_var = tf.placeholder(name='targets', dtype=tf.float32)
    for layer in self.layers:
      layer.build(is_train)


    self.build_flag = True




  def map(self, inputs):

    outputs = self.sess.run(self.tf_map(self.inputs_var), feed_dict={self.inputs_var: inputs})

    return outputs

  def tf_map(self, inputs):

    outputs = self.layers[0].apply(inputs)

    return outputs

  def train(self, inputs, targets, steps=100, batch_size=1):
    if not self.build_flag:
      self.build()

    outputs = self.tf_map(self.inputs_var)

    pair_wise_mse_loss = tf.losses.mean_pairwise_squared_error(labels=self.targets_var,
                                                                    predictions=outputs,
                                                                    weights=1.0,
                                                                    scope=None,
                                                                    loss_collection=tf.GraphKeys.LOSSES
                                                                    )

    mse_loss = tf.losses.mean_squared_error(
      self.targets_var,
      outputs,
      weights=1.0,
      scope=None,
      loss_collection=tf.GraphKeys.LOSSES,
      reduction=tf.losses.Reduction.MEAN)

    regression_loss = tf.losses.get_regularization_losses()

    total_loss = mse_loss # pair_wise_mse_loss # + mse_loss + regression_loss

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(total_loss)

    for i in np.arange(steps):
      self.sess.run(tf.global_variables_initializer())
      start = (i*batch_size) % len(inputs)
      end = (start + batch_size)
      _, loss = self.sess.run([train_op, total_loss], feed_dict={self.inputs_var: inputs[start: end],
                                                                 self.targets_var: targets[start: end]} )
      print('iteration %d, loss:' %(i))
      print(loss)


if __name__ == '__main__':
  hparams = {}
  g = tf.Graph()
  with g.as_default():
    with tf.Session() as session:
      tf_mapper = TfMapper(hparams, session)

      inputs = np.random.rand(10,2)
      targets = np.random.rand(10, 1)

      print(inputs.shape)
      print(targets.shape)
      tf_mapper.train(inputs, targets)