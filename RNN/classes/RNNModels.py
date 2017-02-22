import tensorflow as tf

def data_type():
  return tf.float32


class Basic_LSTM_Model(object):
  """The recurrent model with LSTM and Dropout."""

  def __init__(self, is_training, config, input_):
    self.is_training = is_training
    self._input = input_
    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # LSTM =====================================================================================
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
    # ===================================================================================== LSTM

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = [tf.squeeze(input_step, [1])
    #           for input_step in tf.split(1, num_steps, inputs)]
    # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b

    if not is_training: # ***** experimental feature *****
        self.seq_outputs = tf.nn.softmax(logits)

    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class Basic_RNN_Model(object):
  """The simple recurrent model."""

  def __init__(self, is_training, config, input_, BPTT=True):
      self.is_training = is_training
      self._input = input_
      batch_size = input_.batch_size
      num_steps = input_.num_steps
      size = config.hidden_size
      vocab_size = config.vocab_size

      # SRN / ======================================================================================
      srn_cell = tf.contrib.rnn.BasicRNNCell(size, None, tf.nn.sigmoid)

      if is_training and config.keep_prob < 1:
          srn_cell = tf.nn.rnn_cell.DropoutWrapper(
              srn_cell, output_keep_prob=config.keep_prob)
      cell = tf.nn.rnn_cell.MultiRNNCell([srn_cell] * config.num_layers, state_is_tuple=False)
      # ====================================================================================== / SRN

      self._initial_state = cell.zero_state(batch_size, data_type())

      with tf.device("/cpu:0"):
          embedding = tf.get_variable(
              "embedding", [vocab_size, size], dtype=data_type())
          inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

      if is_training and config.keep_prob < 1:
          inputs = tf.nn.dropout(inputs, config.keep_prob)

      hid_states = []
      targ_list = [] # ***** experimental feature *****
      state = self._initial_state
      with tf.variable_scope("RNN"):
          for time_step in range(num_steps):
              if time_step > 0: tf.get_variable_scope().reuse_variables()
              (cell_act, newstate) = cell(inputs[:, time_step, :], state)
              targ_list.append(input_.targets) # ***** experimental feature *****
              if not BPTT:
                  state = tf.stop_gradient(newstate)  # SRN
              else:
                  state = newstate  # BPTT
              hid_states.append(cell_act)

      output = tf.reshape(tf.concat(axis=1, values=hid_states), [-1, size])
      softmax_w = tf.get_variable(
          "softmax_w", [size, vocab_size], dtype=data_type())
      softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
      logits = tf.matmul(output, softmax_w) + softmax_b
      if not is_training: # ***** experimental feature *****
          self.seq_outputs = tf.nn.softmax(logits)

      # ------------- SPARSE SOFTMAX CROSS ENTROPY WITH LOGITS ----------------
      labels = tf.reshape(input_.targets, [-1])
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

      # ---------------- SIGMOID CROSS ENTROPY WITH LOGITS --------------------
      # labels = tf.reshape(tf.concat(1, targ_list), [-1, vocab_size])
      # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)

      self._cost = cost = tf.reduce_sum(loss) / batch_size
      self._final_state = state

      if not is_training:
          return

      self._lr = tf.Variable(0.0, trainable=False)
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                        config.max_grad_norm)
      optimizer = tf.train.GradientDescentOptimizer(self._lr)
      self._train_op = optimizer.apply_gradients(
          zip(grads, tvars),
          global_step=tf.contrib.framework.get_or_create_global_step())

      self._new_lr = tf.placeholder(
          tf.float32, shape=[], name="new_learning_rate")
      self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
      session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
      return self._input

  @property
  def initial_state(self):
      return self._initial_state

  @property
  def cost(self):
      return self._cost

  @property
  def final_state(self):
      return self._final_state

  @property
  def lr(self):
      return self._lr

  @property
  def train_op(self):
      return self._train_op