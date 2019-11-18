import numpy as np
import random
import math


def dist(p1, p2):
    d = p1 - p2
    return d[0]*d[0]+d[1]*d[1]+d[2]*d[2]


class ReservoirLayer:
    def __init__(self, n_inputs, n_outputs, n_steps, dim, tau_m=64, tau_s=8,
                 threshold=20, refrac=2, weight_scale=1, weight_limit=8, is_input=False,
                 n_input_connect=32, dtype=np.float32):
        self.dtype = dtype
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dim = dim
        self.decay_m = 0 if tau_m == 0 else float(np.exp(-1 / tau_m))
        self.decay_s = 0 if tau_s == 0 else float(np.exp(-1 / tau_s))
        self.threshold = threshold
        self.refrac = refrac
        self.weight_scale = weight_scale
        self.weight_limit = weight_limit
        self.n_steps = n_steps

        self.w = np.zeros((self.n_inputs, self.n_outputs), dtype=self.dtype)
        self.init_input(n_input_connect, is_input)

        self.w_r = np.zeros((self.n_outputs, self.n_outputs), dtype=self.dtype)
        self.init_reservoir()
        self.v = np.zeros(self.n_outputs, dtype=self.dtype)
        self.syn = np.zeros(self.n_outputs, dtype=self.dtype)
        self.pre_out = np.zeros(self.n_outputs, dtype=self.dtype)

    def reset(self):
        self.v = np.zeros(self.n_outputs, dtype=self.dtype)
        self.syn = np.zeros(self.n_outputs, dtype=self.dtype)

    def init_input(self, num, is_input):
        if is_input:
            for pre in range(self.n_inputs):
                for j in range(num):
                    post = random.randrange(self.n_outputs)
                    self.w[pre][post] = random.uniform(-1, 1) * self.weight_scale * self.weight_limit
        else:
            self.w = np.random.default_rng().normal(0.0, self.weight_scale / np.sqrt(self.n_inputs),
                                                    size=(self.n_inputs, self.n_outputs))

    def init_reservoir(self):
        assert self.dim[0] * self.dim[1] * self.dim[2] == self.n_outputs
        excitatoty = np.random.rand(self.n_outputs)
        excitatoty[excitatoty < 0.2] = 0
        excitatoty[excitatoty >= 0.2] = 1
        p = []
        factor1 = 1.5
        factor2 = 4
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                for k in range(self.dim[2]):
                    p.append(np.asarray([i, j, k]))
        for i in range(self.n_outputs):
            for j in range(self.n_outputs):
                if i == j:
                    continue
                if excitatoty[i] == 1:
                    if excitatoty[j] == 1:
                        prob = 0.3 * factor1
                        val = 1
                    else:
                        prob = 0.2 * factor1
                        val = 1
                else:
                    if excitatoty[j] == 1:
                        prob = 0.4 * factor1
                        val = -1
                    else:
                        prob = 0.2 * factor1
                        val = -1
                d = dist(p[i], p[j])
                r = random.random()
                if r < prob * math.exp(-d/factor2):
                    self.w_r[i][j] = val*self.weight_scale

    def forward(self, inputs):
        h1 = np.matmul(inputs, self.w)
        outputs = []
        ref = np.zeros(self.n_outputs)
        for t in range(self.n_steps):
            h_r = np.matmul(self.pre_out, self.w_r)
            self.syn = self.decay_s * self.syn + h1[t, :] + h_r
            self.v = self.decay_m * self.v + self.syn

            self.v[ref > 0] = 0
            ref[ref > 0] = ref[ref > 0] - 1
            v_thr = self.v - self.threshold
            out = np.zeros(self.n_outputs, dtype=self.dtype)
            out[v_thr > 0] = 1.0
            outputs.append(out)
            self.pre_out = out
            ref[v_thr > 0] = self.refrac

            self.v[v_thr > 0] = 0
        outputs = np.stack(outputs)
        return outputs
