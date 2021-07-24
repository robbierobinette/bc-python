import tensorflow as tf
from .Tensor import Tensor
from datetime import datetime


class ElectionModel(tf.keras.Model):
    def __init__(self, n_bins: int, width: int, n_layers: int):
        super(ElectionModel, self).__init__()
        self.n_bins = n_bins
        self.width = width
        self.n_layers = n_layers

        self.my_layers = []
        for i in range(n_layers):
            self.my_layers.append(tf.keras.layers.Dense(width, activation='relu'))

        self.dropout = tf.keras.layers.Dropout(.3)
        self.output_layer = tf.keras.layers.Dense(n_bins, name="action_logits")
        self.softmax = tf.keras.layers.Softmax()

    def call(self, input_data: Tensor, training: bool = None, mask: Tensor = None) -> Tensor:
        tip = input_data
        for layer in self.my_layers:
            tip = layer(tip)
            tip = self.dropout(tip, training=training)
        logits = self.output_layer(tip)
        probability = self.softmax(logits)
        return probability


class ElectionModelTrainer:
    def __init__(self, model: ElectionModel):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam()

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'logs/' + current_time + '/train'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.model_path = ""
        self.global_step = 0

    @staticmethod
    def cross_entropy(y: Tensor, p: Tensor, mask: Tensor) -> Tensor:
        # implement cross-entropy loss, but mask the loss to the one choice that was made
        return - (y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p)) * mask

    def update(self, input_data: Tensor, actions: Tensor, winners: Tensor):
        with tf.GradientTape() as tape:
            probabilities = self.model(input_data)       # shape is (batch_size, n_bins)
            loss = tf.reduce_mean(self.cross_entropy(winners, probabilities, actions))

        grads = tape.gradient(loss, self.model.variables)
        self.optimizer.apply_gradients(zip(grads, self.model.variables))
        self.global_step += 1

        with self.summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.global_step)

        return loss
