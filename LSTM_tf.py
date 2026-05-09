import numpy as np
import tensorflow as tf

# TensorFlow LSTM model 
class LSTM:
    def __init__(self,input_size,hidden_size,output_size,learning_rate=5e-5,beta1=0.9,beta2=0.999,epsilon=1e-8,grad_clip=5.0,weight_decay=1e-5):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.lr = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.weight_decay = float(weight_decay)

        self.cell = tf.keras.layers.LSTMCell(self.hidden_size, dtype=tf.float32)
        self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=True, dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr,
            beta_1=self.beta1,
            beta_2=self.beta2,
            epsilon=self.epsilon,
        )

        self._built = False
        self.Why = np.zeros((self.output_size, self.hidden_size), dtype=np.float32)
        self.by = np.zeros((self.output_size, 1), dtype=np.float32)

    @staticmethod
    def _to_tensor(X):
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return tf.convert_to_tensor(arr, dtype=tf.float32)

    @staticmethod
    def _to_col(target):
        arr = np.asarray(target, dtype=np.float32)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def _ensure_built(self):
        if self._built:
            return
        x0 = tf.zeros((1, self.input_size), dtype=tf.float32)
        h0 = tf.zeros((1, self.hidden_size), dtype=tf.float32)
        c0 = tf.zeros((1, self.hidden_size), dtype=tf.float32)
        h1, [h1s, c1s] = self.cell(x0, states=[h0, c0], training=False)
        _ = self.output_layer(h1s)
        self._built = True
        self._sync_output_weights()

    def _sync_output_weights(self):
        k, b = self.output_layer.get_weights()
        # Keras kernel is (hidden, output). Old code expects (output, hidden).
        self.Why = np.asarray(k.T, dtype=np.float32)
        self.by = np.asarray(b.reshape(-1, 1), dtype=np.float32)

    def _roll_hidden_states(self, X_tf, training=False):
        seq_len = int(X_tf.shape[0])
        h = tf.zeros((1, self.hidden_size), dtype=tf.float32)
        c = tf.zeros((1, self.hidden_size), dtype=tf.float32)
        hs = []
        xs = []
        for t in range(seq_len):
            x_t = tf.reshape(X_tf[t], (1, self.input_size))
            h, [h, c] = self.cell(x_t, states=[h, c], training=training)
            hs.append(h)
            xs.append(x_t)
        return hs, xs, h, c

    def forward(self, X):
        self._ensure_built()
        X_tf = self._to_tensor(X)
        hs, xs, _h_last, _c_last = self._roll_hidden_states(X_tf, training=False)
        cache = []
        for h_t, x_t in zip(hs, xs):
            h_np = np.asarray(tf.transpose(h_t).numpy(), dtype=np.float32)  # (hidden, 1)
            x_np = np.asarray(tf.transpose(x_t).numpy(), dtype=np.float32)  # (input, 1)
            cache.append((h_np, x_np))
        self._sync_output_weights()
        return cache

    def train_step(self, X, target):
        self._ensure_built()
        X_tf = self._to_tensor(X)
        target_col = self._to_col(target)
        y_true = tf.reshape(tf.convert_to_tensor(target_col, dtype=tf.float32), (1, self.output_size))

        with tf.GradientTape() as tape:
            hs, _xs, h_last, _c_last = self._roll_hidden_states(X_tf, training=True)
            y_pred = self.output_layer(h_last)
            mse = tf.reduce_mean(tf.square(y_pred - y_true))
            l2 = tf.add_n([tf.reduce_sum(tf.square(v)) for v in self.cell.trainable_variables + self.output_layer.trainable_variables])
            loss = mse + self.weight_decay * l2

        vars_all = self.cell.trainable_variables + self.output_layer.trainable_variables
        grads = tape.gradient(loss, vars_all)
        if self.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        self.optimizer.apply_gradients(zip(grads, vars_all))
        self._sync_output_weights()
        return float(loss.numpy())

    def save(self, filepath):
        self._ensure_built()
        cell_w = self.cell.get_weights()
        out_w = self.output_layer.get_weights()
        np.savez_compressed(
            filepath,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            grad_clip=0.0 if self.grad_clip is None else self.grad_clip,
            weight_decay=self.weight_decay,
            cell_kernel=cell_w[0],
            cell_recurrent_kernel=cell_w[1],
            cell_bias=cell_w[2],
            out_kernel=out_w[0],
            out_bias=out_w[1],
        )

    @classmethod
    def load(cls, filepath):
        data = np.load(filepath, allow_pickle=False)
        model = cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            output_size=int(data["output_size"]),
            learning_rate=float(data["learning_rate"]),
            beta1=float(data["beta1"]),
            beta2=float(data["beta2"]),
            epsilon=float(data["epsilon"]),
            grad_clip=float(data["grad_clip"]) if float(data["grad_clip"]) != 0.0 else None,
            weight_decay=float(data["weight_decay"]),
        )
        model._ensure_built()
        model.cell.set_weights(
            [
                np.asarray(data["cell_kernel"], dtype=np.float32),
                np.asarray(data["cell_recurrent_kernel"], dtype=np.float32),
                np.asarray(data["cell_bias"], dtype=np.float32),
            ]
        )
        model.output_layer.set_weights(
            [
                np.asarray(data["out_kernel"], dtype=np.float32),
                np.asarray(data["out_bias"], dtype=np.float32),
            ]
        )
        model._sync_output_weights()
        return model

