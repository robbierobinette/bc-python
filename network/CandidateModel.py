from Candidate import Candidate
import numpy as np
import tensorflow as tf
from typing import List
from ActionMemory import ActionMemory
import elections as e
import random
import datetime as datetime
from util.Timings import Timings
from network.Tensor import Tensor
import pickle


class CandidateModel:
    def __init__(self,
                 ideology_bins: int,
                 ideology_dim: int,
                 n_hidden: int,
                 n_latent: int,
                 learn_rate: float):
        super().__init__()
        self.ideology_bins = ideology_bins
        self.ideology_dim = ideology_dim
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.learn_rate = learn_rate

        self.memory = ActionMemory(1024, ideology_dim, ideology_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        self.global_step = 0
        self.memory = ActionMemory(1024, ideology_dim, ideology_dim)
        self.action_width = self.ideology_bins * self.ideology_dim
        self.ideology_range = 2
        self.bin_width = self.ideology_range / self.ideology_bins
        print("bin_width % .3f" % self.bin_width)

        # this is the dimension of the input vector for a single opponent.  It can be the same as ideology_dim, or
        # it could be ideology_dim * ideology_bins for a one_hot representation of ideology
        self.input_width = ideology_bins * ideology_dim

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'logs/' + current_time + '/train'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.model_path = ""




class CandidateModelOld:
    def __init__(self,
                 ideology_bins: int,
                 ideology_dim: int,
                 n_hidden: int,
                 n_latent: int,
                 learn_rate: float):
        super().__init__()
        self.ideology_bins = ideology_bins
        self.ideology_dim = ideology_dim
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.learn_rate = learn_rate
        self.model = CandidateNetwork(ideology_bins=ideology_bins,
                                      ideology_dim=ideology_dim,
                                      n_latent=n_latent,
                                      width=n_hidden)


    def save_to_file(self, path: str):
        self.model_path = path + ".model"
        self.model.save(self.model_path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the model
        del state["model"]
        del state["memory"]
        del state["optimizer"]
        del state["summary_writer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.memory = ActionMemory(100 * 1000, self.ideology_dim, self.ideology_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)
        self.model = tf.keras.models.load_model(self.model_path)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def ready(self) -> bool:
        return self.memory.ready()

    def train(self, batch_size: int):
        for depth in self.memory.depths():
            state, action, reward = self.memory.get_batch(depth, batch_size)
            self.update(state, action, reward)
        self.global_step += 1

    def update(self, input_batch: np.ndarray, actions: np.ndarray, reward: np.ndarray):
        batch_size = input_batch.shape[0]
        one_hot = tf.one_hot(actions, depth=self.ideology_bins)
        # flatten the one_hot array out to match the output of the network
        # each row will have 'ideology_dim' hot elements, one in each chunk of 'ideology_bins'
        one_hot = tf.reshape(one_hot, shape=(batch_size, self.action_width))
        with tf.GradientTape() as tape:
            y = self.model(input_batch, training=True)
            rewards = tf.ones(shape=(batch_size, self.action_width)) * reward
            deltas = tf.square(y - rewards) * one_hot
            loss = tf.reduce_sum(deltas)

        with self.summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.global_step)

        grads = tape.gradient(loss, self.model.variables)
        filtered_grad_vars = [(grad, var) for (grad, var) in zip(grads, self.model.trainable_variables) if
                              grad is not None]
        self.optimizer.apply_gradients(filtered_grad_vars, self.global_step)

    def convert_ideology_to_input(self, ideology: e.Ideology) -> Tensor:
        return self.convert_ideology_to_input_vec(ideology)

    def convert_ideology_to_input_vec(self, ideology: e.Ideology) -> Tensor:
        return ideology.vec.astype(dtype=np.float32)

    def convert_ideology_to_input_onehot(self, ideology: e.Ideology) -> Tensor:
        float_vec = (ideology.vec / self.ideology_range + .5) * self.ideology_bins
        one_hot = tf.one_hot(tf.cast(float_vec, tf.dtypes.int32), depth=self.ideology_bins)
        return tf.reshape(one_hot, shape=(self.input_width))

    def convert_ideology_to_int(self, ideology: float):
        return int((ideology + self.ideology_range / 2) / self.ideology_range * self.ideology_bins)

    # the action vector is a vector of integers corresponding to the actions
    # taken where each action is a location on the i'th dimension of the
    # ideological spectrum
    # i.e.  an ideology of [0,0,0] would correspond to [100, 100, 100]
    #
    def convert_ideology_to_action_vec(self, ideology: e.Ideology) -> Tensor:
        ii = [self.convert_ideology_to_int(i) for i in ideology.vec]
        return tf.constant(ii, dtype=tf.dtypes.int32)

    def get_state_from_opponents(self, opponents: List[Candidate]) -> Tensor:
        # shape is (observation_id, ideology_representation_vec)
        if len(opponents) != 0:
            candidate_observations = [self.convert_ideology_to_input(o.ideology) for o in opponents]
            state = np.stack(candidate_observations)
        else:
            state = tf.zeros(shape=(0, self.input_width), dtype=tf.dtypes.float32)

        return tf.expand_dims(state, 0)

    def add_sample_from_candidates(self, candidate: Candidate, opponents: List[Candidate], winner: Candidate):
        state = self.get_state_from_opponents(opponents)

        action = self.convert_ideology_to_action_vec(candidate.ideology)
        action = tf.expand_dims(action, 0)

        if winner == candidate:
            reward = tf.ones(shape=(1, 1), dtype=tf.dtypes.float32)
        else:
            reward = tf.zeros(shape=(1, 1), dtype=tf.dtypes.float32)

        self.memory.add_sample(state, action, reward)

    def choose_ideology(self, opponents: List[Candidate]) -> Tensor:
        state = self.get_state_from_opponents(opponents)
        ideology_pred = self.model.call(state, training=True)
        ideology_hot = tf.reshape(ideology_pred, shape=(self.ideology_dim, self.ideology_bins))
        ideology_indices = tf.cast(tf.argmax(ideology_hot, axis=1), tf.dtypes.float32)
        ideology_vec = (ideology_indices / self.ideology_bins - .5) * self.ideology_range
        ideology_vec = ideology_vec + tf.random.normal((self.ideology_dim,), 0, self.bin_width / 2)

        return ideology_vec.numpy()