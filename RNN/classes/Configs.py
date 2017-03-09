
class Configs(object):
    def __init__(self, batch_size, hidden_size, init_scale, keep_prob,
                 learning_rate, lr_decay, max_epoch, max_grad_norm, model,
                 max_max_epoch, num_layers, num_steps, vocab_size):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.init_scale = init_scale
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.max_epoch = max_epoch
        self.max_grad_norm = max_grad_norm
        self.max_max_epoch = max_max_epoch
        self.model = model
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.vocab_size = vocab_size

    def clone(self):
        return Configs(self.batch_size, self.hidden_size, self.init_scale, self.keep_prob,  self.learning_rate, self.lr_decay, self.max_epoch,
                    self.max_grad_norm, self.max_max_epoch, self.model,  self.num_layers,  self.num_steps, self.vocab_size)