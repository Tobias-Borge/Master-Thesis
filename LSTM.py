
# Long Short Term Memory
import numpy as np

#Simple (downsampled) sample entropy estimate.
def sample_entropy(signal, m=2, r_ratio=0.2, step=5):
    #Higher values -> more irregular / higher effective frequency.
    x = np.asarray(signal, dtype=float)[::step]
    N = len(x)
    # If the signal is too short, return NaN
    if N <= m + 2:
        return np.nan

    # Compute the threshold for the sample entropy
    r = r_ratio * np.std(x)
    # If the threshold is 0, return 0
    if r == 0.0:
        return 0.0

    # Compute the sample entropy
    def _phi(order):
        # Count the number of matches
        count = 0
        # Loop through the signal
        for i in range(N - order):
            # Loop through the signal
            for j in range(i + 1, N - order):
                # If the difference is less than the threshold, increment the count
                if np.all(np.abs(x[i : i + order] - x[j : j + order]) <= r):
                    count += 1
        return count

    # Compute the sample entropy
    B = _phi(m)
    # Compute the sample entropy
    A = _phi(m + 1)
    # If the sample entropy is 0, return 0
    if B == 0 or A == 0:
        return 0.0
    return -np.log(A / B)

# LSTM model
class LSTM:
    # Initialize the LSTM model
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-8, grad_clip=5.0, weight_decay=1e-5):
        # Initialize the LSTM model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        
        # Xavier/Glorot initialization for the LSTM weights (initialize the LSTM weights)
        fan_in = hidden_size + input_size
        fan_out = hidden_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))

        # LSTM weight matrices (forget, input, cell, output gates)
        self.Wf = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))
        self.Wi = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))
        self.Wc = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))
        self.Wo = np.random.uniform(-limit, limit, (hidden_size, hidden_size + input_size))

        # Bias vectors (initialize forget gate bias to 1 for better gradient flow) (bias for the LSTM gates)
        self.bf = np.ones((hidden_size, 1))  # Start with 1 to help remember initially
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        # Linear output layer: maps hidden state to asset returns (output layer)
        limit_out = np.sqrt(6.0 / (hidden_size + output_size))
        self.Why = np.random.uniform(-limit_out, limit_out, (output_size, hidden_size))
        self.by = np.zeros((output_size, 1))
        
        # Adam optimizer state
        self.t = 0  # Time step counter
        self.m_Wf = np.zeros_like(self.Wf)
        self.m_Wi = np.zeros_like(self.Wi)
        self.m_Wc = np.zeros_like(self.Wc)
        self.m_Wo = np.zeros_like(self.Wo)
        self.m_bf = np.zeros_like(self.bf)
        self.m_bi = np.zeros_like(self.bi)
        self.m_bc = np.zeros_like(self.bc)
        self.m_bo = np.zeros_like(self.bo)
        self.m_Why = np.zeros_like(self.Why)
        self.m_by = np.zeros_like(self.by)
        
        # Second moment estimates (squared gradients)
        self.v_Wf = np.zeros_like(self.Wf)
        self.v_Wi = np.zeros_like(self.Wi)
        self.v_Wc = np.zeros_like(self.Wc)
        self.v_Wo = np.zeros_like(self.Wo)
        self.v_bf = np.zeros_like(self.bf)
        self.v_bi = np.zeros_like(self.bi)
        self.v_bc = np.zeros_like(self.bc)
        self.v_bo = np.zeros_like(self.bo)
        self.v_Why = np.zeros_like(self.Why)
        self.v_by = np.zeros_like(self.by)

    # Activation functions
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)

    @staticmethod
    def dtanh(x):
        return 1 - np.tanh(x)**2

    # Forward pass
    def forward(self, X):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        cache = []

        for x_t in X:
            x_t = x_t.reshape(-1, 1)
            concat = np.vstack((h, x_t))

            f_t = self.sigmoid(self.Wf @ concat + self.bf)
            i_t = self.sigmoid(self.Wi @ concat + self.bi)
            c_hat = np.tanh(self.Wc @ concat + self.bc)
            c = f_t * c + i_t * c_hat
            o_t = self.sigmoid(self.Wo @ concat + self.bo)
            h = o_t * np.tanh(c)

            cache.append((h, c, f_t, i_t, c_hat, o_t, concat))

        return cache

    # Clip gradients to prevent exploding gradients
    def _clip_gradients(self, *grads):

        if self.grad_clip is None:
            return grads
        
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-8)
            return [g * scale for g in grads]
        return grads
    
    # Update the parameters using the Adam optimizer
    def _adam_update(self, param, grad, m, v):

        # Update biased first moment estimate
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        
        # Update biased second raw moment estimate
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        param[:] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    # Backward pass
    def backward(self, cache, dLdh_last):
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)

        dh_next = dLdh_last
        dc_next = np.zeros((self.hidden_size, 1))
 
        # Backward pass through the LSTM
        for t in reversed(range(len(cache))):
            h, c, f_t, i_t, c_hat, o_t, concat = cache[t]
            c_prev = cache[t-1][1] if t > 0 else np.zeros_like(c)

            do = dh_next * np.tanh(c) * o_t * (1 - o_t)
            dc = dh_next * o_t * (1 - np.tanh(c)**2) + dc_next

            df = dc * c_prev * f_t * (1 - f_t)
            di = dc * c_hat * i_t * (1 - i_t)
            dc_hat = dc * i_t * (1 - c_hat**2)

            dWf += df @ concat.T
            dWi += di @ concat.T
            dWc += dc_hat @ concat.T
            dWo += do @ concat.T

            dbf += df
            dbi += di
            dbc += dc_hat
            dbo += do

            dconcat = (
                self.Wf.T @ df +
                self.Wi.T @ di +
                self.Wc.T @ dc_hat +
                self.Wo.T @ do
            )

            dh_next = dconcat[:self.hidden_size, :]
            dc_next = dc * f_t

        # Clip gradients to prevent exploding gradients
        dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo = self._clip_gradients(
            dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo
        )

        # Update LSTM weights using Adam optimizer
        self.t += 1
        self._adam_update(self.Wf, dWf, self.m_Wf, self.v_Wf)
        self._adam_update(self.Wi, dWi, self.m_Wi, self.v_Wi)
        self._adam_update(self.Wc, dWc, self.m_Wc, self.v_Wc)
        self._adam_update(self.Wo, dWo, self.m_Wo, self.v_Wo)
        self._adam_update(self.bf, dbf, self.m_bf, self.v_bf)
        self._adam_update(self.bi, dbi, self.m_bi, self.v_bi)
        self._adam_update(self.bc, dbc, self.m_bc, self.v_bc)
        self._adam_update(self.bo, dbo, self.m_bo, self.v_bo)

    # Single training step
    def train_step(self, X, target):

        cache = self.forward(X)
        h_last = cache[-1][0]  # last hidden state

        # Linear output layer: maps hidden state to asset returns (output layer)
        y_pred = self.Why @ h_last + self.by

        # Compute mean squared error loss with L2 regularization (MSE loss)
        mse_loss = np.mean((y_pred - target)**2)
        # L2 regularization penalty
        l2_penalty = self.weight_decay * (
            np.sum(self.Wf**2) + np.sum(self.Wi**2) + np.sum(self.Wc**2) + np.sum(self.Wo**2) +
            np.sum(self.Why**2)
        )
        # Total loss
        loss = mse_loss + l2_penalty

        # Gradient of loss with respect to output
        dLdy = 2 * (y_pred - target) / self.output_size

        # Backprop through linear layer
        dLdh_last = self.Why.T @ dLdy
        self.backward(cache, dLdh_last)

        # Gradients for output layer
        dWhy = dLdy @ h_last.T
        dby = dLdy
        
        # Clip gradients for output layer
        if self.grad_clip is not None:
            total_norm = np.sqrt(np.sum(dWhy**2) + np.sum(dby**2))
            if total_norm > self.grad_clip:
                scale = self.grad_clip / (total_norm + 1e-8)
                dWhy *= scale
                dby *= scale

        # Update linear output weights using Adam optimizer
        self._adam_update(self.Why, dWhy, self.m_Why, self.v_Why)
        self._adam_update(self.by, dby, self.m_by, self.v_by)

        return loss
    
    # Save the LSTM model to a file
    def save(self, filepath):

        np.savez_compressed(
            filepath,
            # Model architecture parameters
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            grad_clip=self.grad_clip if self.grad_clip is not None else 0.0,
            weight_decay=self.weight_decay,
            # LSTM weights
            Wf=self.Wf,
            Wi=self.Wi,
            Wc=self.Wc,
            Wo=self.Wo,
            # LSTM biases
            bf=self.bf,
            bi=self.bi,
            bc=self.bc,
            bo=self.bo,
            # Output layer weights
            Why=self.Why,
            by=self.by,
            # Adam optimizer state
            t=self.t,
            m_Wf=self.m_Wf,
            m_Wi=self.m_Wi,
            m_Wc=self.m_Wc,
            m_Wo=self.m_Wo,
            m_bf=self.m_bf,
            m_bi=self.m_bi,
            m_bc=self.m_bc,
            m_bo=self.m_bo,
            m_Why=self.m_Why,
            m_by=self.m_by,
            v_Wf=self.v_Wf,
            v_Wi=self.v_Wi,
            v_Wc=self.v_Wc,
            v_Wo=self.v_Wo,
            v_bf=self.v_bf,
            v_bi=self.v_bi,
            v_bc=self.v_bc,
            v_bo=self.v_bo,
            v_Why=self.v_Why,
            v_by=self.v_by
        )
    
    # Load the LSTM model from a file
    @classmethod
    def load(cls, filepath):

        data = np.load(filepath)
        
        # Create LSTM instance with saved parameters
        lstm = cls(
            input_size=int(data['input_size']),
            hidden_size=int(data['hidden_size']),
            output_size=int(data['output_size']),
            learning_rate=float(data['learning_rate']),
            beta1=float(data['beta1']),
            beta2=float(data['beta2']),
            epsilon=float(data['epsilon']),
            grad_clip=float(data['grad_clip']) if 'grad_clip' in data and data['grad_clip'] != 0 else None,
            weight_decay=float(data['weight_decay'])
        )
        
        # Load weights and biases
        lstm.Wf = data['Wf']
        lstm.Wi = data['Wi']
        lstm.Wc = data['Wc']
        lstm.Wo = data['Wo']
        lstm.bf = data['bf']
        lstm.bi = data['bi']
        lstm.bc = data['bc']
        lstm.bo = data['bo']
        lstm.Why = data['Why']
        lstm.by = data['by']
        
        # Load Adam optimizer state
        lstm.t = int(data['t'])
        lstm.m_Wf = data['m_Wf']
        lstm.m_Wi = data['m_Wi']
        lstm.m_Wc = data['m_Wc']
        lstm.m_Wo = data['m_Wo']
        lstm.m_bf = data['m_bf']
        lstm.m_bi = data['m_bi']
        lstm.m_bc = data['m_bc']
        lstm.m_bo = data['m_bo']
        lstm.m_Why = data['m_Why']
        lstm.m_by = data['m_by']
        lstm.v_Wf = data['v_Wf']
        lstm.v_Wi = data['v_Wi']
        lstm.v_Wc = data['v_Wc']
        lstm.v_Wo = data['v_Wo']
        lstm.v_bf = data['v_bf']
        lstm.v_bi = data['v_bi']
        lstm.v_bc = data['v_bc']
        lstm.v_bo = data['v_bo']
        lstm.v_Why = data['v_Why']
        lstm.v_by = data['v_by']
        
        return lstm

