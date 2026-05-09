import numpy as np
import tensorflow as tf

# TensorFlow MLP model
class MLP:
    # Initialize the MLP model

    def __init__(self,input_size,hidden_size,output_size,learning_rate=5e-5,beta1=0.9,beta2=0.999,epsilon=1e-8,grad_clip=5.0,weight_decay=1e-5,):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.lr = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.weight_decay = float(weight_decay)

        self.hidden_layer = tf.keras.layers.Dense(self.hidden_size, activation="tanh", use_bias=True, dtype=tf.float32)
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
            arr = arr.reshape(1, -1)
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
        h0 = self.hidden_layer(x0)
        _ = self.output_layer(h0)
        self._built = True
        self._sync_output_weights()

    def _sync_output_weights(self):
        k, b = self.output_layer.get_weights()
        self.Why = np.asarray(k.T, dtype=np.float32)
        self.by = np.asarray(b.reshape(-1, 1), dtype=np.float32)

    def forward(self, X):
        self._ensure_built()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        cache = []
        for row in X_arr:
            x_t = row.reshape(1, -1)
            h_t = self.hidden_layer(tf.convert_to_tensor(x_t, dtype=tf.float32)).numpy().reshape(-1, 1)
            cache.append((h_t.astype(np.float32), row.reshape(-1, 1).astype(np.float32)))
        self._sync_output_weights()
        return cache

    def train_step(self, X, target):
        self._ensure_built()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        x_last = tf.convert_to_tensor(X_arr[-1].reshape(1, -1), dtype=tf.float32)
        target_col = self._to_col(target)
        y_true = tf.reshape(tf.convert_to_tensor(target_col, dtype=tf.float32), (1, self.output_size))

        with tf.GradientTape() as tape:
            h = self.hidden_layer(x_last, training=True)
            y_pred = self.output_layer(h, training=True)
            mse = tf.reduce_mean(tf.square(y_pred - y_true))
            l2 = tf.add_n([tf.reduce_sum(tf.square(v)) for v in self.hidden_layer.trainable_variables + self.output_layer.trainable_variables])
            loss = mse + self.weight_decay * l2

        vars_all = self.hidden_layer.trainable_variables + self.output_layer.trainable_variables
        grads = tape.gradient(loss, vars_all)
        if self.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        self.optimizer.apply_gradients(zip(grads, vars_all))
        self._sync_output_weights()
        return float(loss.numpy())

    def save(self, filepath):
        self._ensure_built()
        hid_w = self.hidden_layer.get_weights()
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
            hid_kernel=hid_w[0],
            hid_bias=hid_w[1],
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
        model.hidden_layer.set_weights(
            [
                np.asarray(data["hid_kernel"], dtype=np.float32),
                np.asarray(data["hid_bias"], dtype=np.float32),
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

