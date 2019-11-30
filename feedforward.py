import torch
import random
import matplotlib.pyplot as plot


def plot_v(v, num):
    for i in range(num):
        f = plot.plot(v[:, i])
    plot.show(f)


class SpikingLayer:
    def __init__(self, device, n_inputs, n_outputs, n_steps, tau_m=64, tau_s=8, tau_c=64, cal_mid=5, cal_margin=3,
                 threshold=15, refrac=2, weight_scale=8, weight_limit=8, is_input=False, n_input_connect=32,
                 delta_pot=0.006, delta_dep=0.006, stdp_i=False, dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.tau_c = tau_c
        self.threshold = threshold
        self.refrac = refrac
        self.weight_scale = weight_scale
        self.n_steps = n_steps
        self.cal_mid = cal_mid
        self.cal_margin = cal_margin
        self.delta_pot = delta_pot
        self.delta_dep = delta_dep
        self.stdp_i = stdp_i
        self.stdp_lambda = 1 / 512
        self.stdp_TAU_X_TRACE = 4
        self.stdp_TAU_Y_TRACE = 8
        self.A_neg = 0.01
        self.A_pos = 0.005
        self.trace_x = torch.zeros(self.n_inputs, dtype=self.dtype, device=self.device)
        self.trace_y = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.weight_limit = weight_limit
        self.w = torch.zeros((self.n_inputs, self.n_outputs), dtype=self.dtype, device=self.device)
        self.init_input(n_input_connect, is_input)

        self.v = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.syn = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.cal = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)

    def reset(self):
        self.v = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.syn = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        self.cal = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)

    def init_input(self, num, is_input):
        if is_input:
            for pre in range(self.n_inputs):
                for j in range(num):
                    post = random.randrange(self.n_outputs)
                    self.w[pre, post] = self.w[pre, post] + random.uniform(-1, 1) * self.weight_scale
        else:
            self.w = torch.rand((self.n_inputs, self.n_outputs), dtype=self.dtype, device=self.device)
            self.w = self.w * 2 - 1

    def forward(self, inputs, epoch=-1, label=-1):
        h1 = torch.matmul(inputs, self.w)
        outputs = []
        v_all = []
        cal_all = []
        ref = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
        for t in range(self.n_steps):
            self.syn = self.syn - self.syn / self.tau_s + h1[t, :]
            self.v = self.v - self.v / self.tau_m + self.syn / self.tau_s
            self.cal = self.cal - self.cal/self.tau_c
            if label >= 0:
                teacher = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
                teacher[label] = 1
                self.v[(teacher == 1) & (self.cal < (self.cal_mid + 1))] = \
                    self.v[(teacher == 1) & (self.cal < (self.cal_mid + 1))] + self.threshold
                self.v[(teacher == 0) & (self.cal > (self.cal_mid - 1))] = \
                    self.v[(teacher == 0) & (self.cal > (self.cal_mid - 1))] - self.threshold * 0.75

            self.v[ref > 0] = 0
            ref[ref > 0] = ref[ref > 0] - 1
            v_all.append(self.v)

            out = torch.zeros(self.n_outputs, dtype=self.dtype, device=self.device)
            out[self.v > self.threshold] = 1.0
            outputs.append(out)

            self.cal[self.v > self.threshold] = self.cal[self.v > self.threshold] + 1
            cal_all.append(self.cal)
            if label >= 0:
                self.calcuim_supervised_rule(epoch, torch.tensor(inputs[t, :], dtype=self.dtype, device=self.device))
            self.v[self.v > self.threshold] = 0
            ref[self.v > self.threshold] = self.refrac

            if self.stdp_i:
                in_s = inputs[t, :]
                self.trace_x = self.trace_x / self.stdp_TAU_X_TRACE
                self.trace_y = self.trace_y / self.stdp_TAU_Y_TRACE
                self.trace_y[out == 1] = self.trace_y[out == 1] + 1
                self.trace_x[in_s == 1] = self.trace_x[in_s == 1] + 1

                m_y = self.trace_y.repeat(self.n_inputs, 1)
                w_tmp = self.A_neg * self.stdp_lambda * m_y
                w_tmp[self.w < 0] = -w_tmp[self.w < 0]
                self.w[in_s == 1, :] = self.w[in_s == 1, :] - w_tmp[in_s == 1, :]

                m_x = torch.transpose(self.trace_x.repeat(self.n_outputs, 1), 0, 1)
                w_tmp = self.A_pos * self.stdp_lambda * m_x
                w_tmp[self.w < 0] = -w_tmp[self.w < 0]
                self.w[:, out == 1] = self.w[:, out == 1] + self.stdp_lambda * w_tmp[:, out == 1]

                self.w[self.w > self.weight_limit] = self.weight_limit
                self.w[self.w < -self.weight_limit] = -self.weight_limit
        outputs = torch.stack(outputs)
        return outputs

    def calcuim_supervised_rule(self, epoch, inputs):

        mask = torch.zeros((self.n_inputs, self.n_outputs), dtype=self.dtype, device=self.device)
        mask[inputs == 1, :] = mask[inputs == 1, :] + 1
        mask[:, (self.cal_mid < self.cal) & (self.cal < (self.cal_mid + self.cal_margin))] = \
            mask[:, (self.cal_mid < self.cal) & (self.cal < (self.cal_mid + self.cal_margin))] + 1
        val = self.delta_pot / (1 + epoch / 25)
        self.w[mask == 2] = self.w[mask == 2] + val

        mask = torch.zeros((self.n_inputs, self.n_outputs), dtype=self.dtype, device=self.device)
        mask[inputs == 1, :] = mask[inputs == 1, :] + 1
        mask[:, ((self.cal_mid - self.cal_margin) < self.cal) & (self.cal < self.cal_mid)] = \
            mask[:, ((self.cal_mid - self.cal_margin) < self.cal) & (self.cal < self.cal_mid)] + 1

        val = self.delta_dep / (1 + epoch / 25)
        self.w[mask == 2] = self.w[mask == 2] - val

        self.w[self.w > self.weight_limit] = self.weight_limit
        self.w[self.w < -self.weight_limit] = -self.weight_limit
