from mapping_models.basic_mapper import BasicMapper
import tensorflow as tf


class Layer(object):
  def __init__(self):
    self.build_flag = False

  def build(self, is_train=True, scope ='', reuse=False, **kwargs):
    raise NotImplementedError()

  def apply(self, inputs):
    raise NotImplementedError


class SingleLinearLayer(Layer):
  def __init__(self, scope):
    super(SingleLinearLayer, self).__init__(hparams)
    self.scope = scope


  def build(self, is_train=True, reuse=False, **kwargs):
    """Create variables and define the graph.

    :param is_train:
    :param kwargs:
    :return:
    """
    with tf.variable_scope('LinearLayer_'+self.scope, reuse=reuse,
                           initializer=tf.initializers.truncated_normal):
      self.W = tf.get_variable('w',
                                  dtype=tf.float32,
                                  trainable=is_train)


  def apply(self, inputs):
    """

    :param inputs: [batch_size, embedding_size]
    :return:
    """
    with tf.variable_scope('LinearLayer_' + self.scope, reuse=tf.AUTO_REUSE,
                           initializer=tf.initializers.truncated_normal,
                           regularizer=tf.contrib.layers.l2_regularizer):
      outputs = tf.matmul(self.W, inputs)

    return outputs



class WeightedInputLinearLayer(Layer):
  def __init__(self,scope, input_dim, output_dim, input_len,activation=tf.nn.relu):
    super(WeightedInputLinearLayer, self).__init__()
    self.scope = scope
    self.activation=activation
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.input_len = input_len

  def build(self, is_train=True, reuse=False, **kwargs):
    """Create variables and define the graph.

    :param is_train:
    :param kwargs:
    :return:
    """

    train_weights = is_train
    if 'train_weights_flag' in kwargs:
      train_weights = kwargs['train_weights_flag']

    with tf.variable_scope('LinearLayer_'+self.scope, reuse=reuse, initializer=tf.initializers.truncated_normal):
      self.W = tf.get_variable(name='w', shape=(self.input_dim, self.output_dim),
                                  dtype=tf.float32,
                                  trainable=is_train)
      self.input_weights = tf.get_variable(name='input_weights', shape=(self.input_len),
                                           dtype=tf.float32,
                                           trainable=train_weights)


  def apply(self, inputs):
    """

    :param inputs: [batch_size, length, embedding_size]
    :return:
    """
    with tf.variable_scope('LinearLayer_' + self.scope, reuse=tf.AUTO_REUSE,
                           initializer=tf.initializers.truncated_normal,
                           regularizer=tf.contrib.layers.l2_regularizer):

      batch_size = tf.shape(inputs)[0]
      input_weights_shape = self.input_weights.get_shape()
      #broad_casted_input_weights = tf.broadcast_to(self.input_weights, shape=[batch_size, *input_weights_shape])
      outputs = tf.matmul(tf.multiply(self.input_weights,inputs), self.W)

    return outputs