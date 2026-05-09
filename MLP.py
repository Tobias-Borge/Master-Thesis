# Multi-layer Perceptron
import numpy as np

#Lightweight MLP with an API compatible with the current LSTM usage:
class MLP:
    #Initialize the MLP model
    def __init__(self,input_size,hidden_size,output_size,learning_rate=5e-5,beta1=0.9,beta2=0.999,epsilon=1e-8,grad_clip=5.0,weight_decay=1e-5):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.lr = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.grad_clip = grad_clip
        self.weight_decay = float(weight_decay)

        limit_h = np.sqrt(6.0 / (self.input_size + self.hidden_size))
        self.Wxh = np.random.uniform(-limit_h, limit_h, (self.hidden_size, self.input_size))
        self.bh = np.zeros((self.hidden_size, 1), dtype=float)

        limit_out = np.sqrt(6.0 / (self.hidden_size + self.output_size))
        self.Why = np.random.uniform(-limit_out, limit_out, (self.output_size, self.hidden_size))
        self.by = np.zeros((self.output_size, 1), dtype=float)

        self.t = 0
        self.m_Wxh = np.zeros_like(self.Wxh)
        self.m_bh = np.zeros_like(self.bh)
        self.m_Why = np.zeros_like(self.Why)
        self.m_by = np.zeros_like(self.by)

        self.v_Wxh = np.zeros_like(self.Wxh)
        self.v_bh = np.zeros_like(self.bh)
        self.v_Why = np.zeros_like(self.Why)
        self.v_by = np.zeros_like(self.by)

    @staticmethod
    def _to_col(x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _dtanh_from_activation(a):
        return 1.0 - a**2

    def _adam_update(self, param, grad, m, v):
        m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
        v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

        m_hat = m / (1.0 - self.beta1**self.t)
        v_hat = v / (1.0 - self.beta2**self.t)
        param[:] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def _clip_gradients(self, *grads):
        if self.grad_clip is None:
            return grads
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-8)
            return [g * scale for g in grads]
        return grads

    #Forward pass
    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Process each row in sequence order and return a list cache
        cache = []
        for row in X:
            x_t = row.reshape(-1, 1)
            # Apply the tanh activation function to the input
            h_t = self._tanh(self.Wxh @ x_t + self.bh)
            cache.append((h_t, x_t))
        return cache

    #Training step
    def train_step(self, X, target):
        cache = self.forward(X)
        h_last, x_last = cache[-1]
        target = self._to_col(target)
        # Linear output layer: maps hidden state to asset returns (output layer)
        y_pred = self.Why @ h_last + self.by
        # Compute mean squared error loss with L2 regularization (MSE loss)
        mse_loss = np.mean((y_pred - target) ** 2)
        # L2 regularization penalty
        l2_penalty = self.weight_decay * (np.sum(self.Wxh**2) + np.sum(self.Why**2))
        loss = float(mse_loss + l2_penalty)

        # Gradient of loss with respect to output
        dLdy = 2.0 * (y_pred - target) / max(self.output_size, 1)
        dWhy = dLdy @ h_last.T + 2.0 * self.weight_decay * self.Why
        dby = dLdy

        # Gradient of loss with respect to hidden state
        dLdh = self.Why.T @ dLdy
        # Gradient of loss with respect to input
        dz = dLdh * self._dtanh_from_activation(h_last)
        dWxh = dz @ x_last.T + 2.0 * self.weight_decay * self.Wxh
        dbh = dz

        # Clip gradients
        dWxh, dbh, dWhy, dby = self._clip_gradients(dWxh, dbh, dWhy, dby)

        self.t += 1
        # Update weights using Adam optimizer
        self._adam_update(self.Wxh, dWxh, self.m_Wxh, self.v_Wxh)
        # Update bias using Adam optimizer
        self._adam_update(self.bh, dbh, self.m_bh, self.v_bh)
        # Update output weights using Adam optimizer
        self._adam_update(self.Why, dWhy, self.m_Why, self.v_Why)
        # Update output bias using Adam optimizer
        self._adam_update(self.by, dby, self.m_by, self.v_by)
        # Return loss
        return loss

    #Save the model
    def save(self, filepath):
        np.savez_compressed(
            filepath,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            grad_clip=self.grad_clip if self.grad_clip is not None else 0.0,
            weight_decay=self.weight_decay,
            Wxh=self.Wxh,
            bh=self.bh,
            Why=self.Why,
            by=self.by,
            t=self.t,
            m_Wxh=self.m_Wxh,
            m_bh=self.m_bh,
            m_Why=self.m_Why,
            m_by=self.m_by,
            v_Wxh=self.v_Wxh,
            v_bh=self.v_bh,
            v_Why=self.v_Why,
            v_by=self.v_by,
        )

    #Load the model
    @classmethod
    def load(cls, filepath):
        data = np.load(filepath)
        mlp = cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            output_size=int(data["output_size"]),
            learning_rate=float(data["learning_rate"]),
            beta1=float(data["beta1"]),
            beta2=float(data["beta2"]),
            epsilon=float(data["epsilon"]),
            grad_clip=float(data["grad_clip"]) if data["grad_clip"] != 0 else None,
            weight_decay=float(data["weight_decay"]),
        )
        mlp.Wxh = data["Wxh"]
        mlp.bh = data["bh"]
        mlp.Why = data["Why"]
        mlp.by = data["by"]

        mlp.t = int(data["t"])
        mlp.m_Wxh = data["m_Wxh"]
        mlp.m_bh = data["m_bh"]
        mlp.m_Why = data["m_Why"]
        mlp.m_by = data["m_by"]
        mlp.v_Wxh = data["v_Wxh"]
        mlp.v_bh = data["v_bh"]
        mlp.v_Why = data["v_Why"]
        mlp.v_by = data["v_by"]
        return mlp














