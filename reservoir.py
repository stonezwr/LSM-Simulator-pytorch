import random
import math
import torch


def dist(p1, p2):
    d = p1 - p2
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


class ReservoirLayer:
    def __init__(self, device, n_inputs, n_outputs, n_steps, dim, tau_m=64, tau_s=8,
                 threshold=15, refrac=2, weight_scale=8, weight_limit=8, is_input=False,
                 n_input_connect=32, homeostasis=False, stdp_r=False, stdp_i=False, dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dim = dim
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.threshold = torch.ones(self.n_outputs, device=self.device, dtype=self.dtype) * threshold
        self.refrac = refrac
        self.weight_scale = weight_scale
        self.weight_limit = weight_limit
        self.n_steps = n_steps
        self.homeostasis = homeostasis
        self.stdp_r = stdp_r
        self.stdp_i = stdp_i
        self.stdp_lambda = 1/512
        self.stdp_TAU_X_TRACE_E = 4
        self.stdp_TAU_X_TRACE_I = 2
        self.stdp_TAU_Y_TRACE_E = 8
        self.stdp_TAU_Y_TRACE_I = 4
        self.A_neg = 0.01
        self.A_pos = 0.005
        self.trace_x_i = torch.zeros(self.n_inputs, dtype=self.dtype, device=self.device)
        self.trace_x_r = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.trace_y = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)

        self.excitatoty = torch.rand(self.n_outputs, dtype=self.dtype, device=self.device)
        self.excitatoty[self.excitatoty < 0.2] = -1
        self.excitatoty[self.excitatoty >= 0.2] = 1
        self.w = torch.zeros((self.n_inputs, self.n_outputs), dtype=self.dtype, device=self.device)
        self.init_input(n_input_connect, is_input)

        self.w_r = torch.zeros((self.n_outputs, self.n_outputs), dtype=self.dtype, device=self.device)
        self.w_r[1, :] = 1

        self.v = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.syn = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.pre_out = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)

    def reset(self):
        self.v = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.syn = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.trace_x_i = torch.zeros(self.n_inputs, dtype=self.dtype, device=self.device)
        self.trace_x_r = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.trace_y = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.pre_out = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)

    def init_input(self, num, is_input):
        if is_input:
            for pre in range(self.n_inputs):
                for j in range(num):
                    post = random.randrange(self.n_outputs)
                    self.w[pre, post] = self.w[pre, post] + random.uniform(-1, 1) * self.weight_scale
        else:
            self.w = torch.rand((self.n_inputs, self.n_outputs), dtype=self.dtype, device=self.device)
            self.w = self.w * 2 - 1

    def init_reservoir(self):
        assert self.dim[0] * self.dim[1] * self.dim[2] == self.n_outputs
        p = []
        factor1 = 1.5
        factor2 = 4
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                for k in range(self.dim[2]):
                    p.append(torch.tensor([i, j, k], dtype=self.dtype, device=self.device))
        for i in range(self.n_outputs):
            for j in range(self.n_outputs):
                if i == j:
                    continue
                if self.excitatoty[i] == 1:
                    if self.excitatoty[j] == 1:
                        prob = 0.3 * factor1
                        val = 1
                    else:
                        prob = 0.2 * factor1
                        val = 1
                else:
                    if self.excitatoty[j] == 1:
                        prob = 0.4 * factor1
                        val = -1
                    else:
                        prob = 0.2 * factor1
                        val = -1
                d = dist(p[i], p[j])
                r = random.random()
                if r < prob * math.exp(-d / factor2):
                    self.w_r[i][j] = val

    def forward(self, inputs):
        h1 = torch.matmul(inputs, self.w)
        outputs = []
        ref = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        for t in range(self.n_steps):
            h_r = torch.matmul(self.pre_out, self.w_r)  # w_r: in*out
            self.syn = self.syn - self.syn / self.tau_s + (h1[t, :] + h_r) * self.excitatoty
            self.v = self.v - self.v / self.tau_m + self.syn / self.tau_s

            self.v[ref > 0] = 0
            ref[ref > 0] = ref[ref > 0] - 1
            v_thr = self.v - self.threshold
            out = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
            out[v_thr > 0] = 1.0
            outputs.append(out)
            self.pre_out = out
            ref[v_thr > 0] = self.refrac
            self.v[v_thr > 0] = 0

            if self.homeostasis:
                self.threshold = self.threshold - self.threshold / 32
                self.threshold[out == 1] = self.threshold[out == 1] + 1
                self.threshold[self.threshold < 8] = 8
                self.threshold[self.threshold > 32] = 32

            if self.stdp_r or self.stdp_i:
                self.trace_y[self.excitatoty == 1] = self.trace_y[self.excitatoty == 1] / self.stdp_TAU_Y_TRACE_E
                self.trace_y[self.excitatoty == -1] = self.trace_y[self.excitatoty == -1] / self.stdp_TAU_Y_TRACE_I
                self.trace_y[out == 1] = self.trace_y[out == 1] + 1

            if self.stdp_r:
                self.trace_x_r[self.excitatoty == 1] = self.trace_x_r[self.excitatoty == 1] / self.stdp_TAU_X_TRACE_E
                self.trace_x_r[self.excitatoty == -1] = self.trace_x_r[self.excitatoty == -1] / self.stdp_TAU_X_TRACE_I
                self.trace_x_r[self.pre_out == 1] = self.trace_x_r[self.pre_out == 1] + 1

                m_y = self.trace_y.repeat(self.n_outputs, 1)
                w_tmp = self.A_neg * self.stdp_lambda * m_y
                w_tmp[self.w_r < 0] = -w_tmp[self.w_r < 0]
                self.w_r[self.pre_out == 1, :] = self.w_r[self.pre_out == 1, :] - w_tmp[self.pre_out == 1, :]

                m_x = self.trace_x_r.repeat(self.n_outputs, 1)
                torch.transpose(m_x, 0, 1)
                w_tmp = self.A_pos * self.stdp_lambda * m_x
                w_tmp[self.w_r < 0] = -w_tmp[self.w_r < 0]
                self.w_r[:, out == 1] = self.w_r[:, out == 1] + w_tmp[:, out == 1]
                self.w_r[self.w_r > self.weight_limit] = self.weight_limit
                self.w_r[self.w_r < -self.weight_limit] = -self.weight_limit

            in_s = inputs[t, :]
            if self.stdp_i and (torch.sum(in_s) > 0 or torch.sum(out) > 0):
                in_s = inputs[t, :]
                self.trace_x_i[in_s == 1] = self.trace_x_i[in_s == 1] + 1
                self.trace_x_i = self.trace_x_i / self.stdp_TAU_X_TRACE_E
                m_y = self.trace_y.repeat(self.n_inputs, 1)
                w_tmp = self.A_neg * self.stdp_lambda * m_y
                w_tmp[self.w < 0] = -w_tmp[self.w < 0]
                self.w[in_s == 1, :] = self.w[in_s == 1, :] - w_tmp[in_s == 1, :]
                m_x = torch.transpose(self.trace_x_i.repeat(self.n_outputs, 1), 0, 1)
                w_tmp = self.A_pos * self.stdp_lambda * m_x
                w_tmp[self.w < 0] = -w_tmp[self.w < 0]
                self.w[:, out == 1] = self.w[:, out == 1] + w_tmp[:, out == 1]

                self.w[self.w > self.weight_limit] = self.weight_limit
                self.w[self.w < -self.weight_limit] = -self.weight_limit
        outputs = torch.stack(outputs)
        return outputs
