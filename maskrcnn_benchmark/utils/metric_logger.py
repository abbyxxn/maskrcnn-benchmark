# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


class TFLogger(MetricLogger):
    def __init__(self, log_period=20, start_iter=0, max_iter=90000, summary_logger=None, delimiter="  "):
        super(TFLogger, self).__init__(delimiter)
        # init summary logger
        self.LOG_PERIOD = log_period
        self.iteration = start_iter
        self.MAX_ITER = max_iter
        self.summary_logger = summary_logger
        self.tb_ignored_keys = ['data']

    def update(self, **kwargs):
        super(TFLogger, self).update(**kwargs)
        self.tb_log_stats(**kwargs)

    def set_iteration(self, iteration):
        self.iteration = iteration

    def tb_log_stats(self, **kwargs):
        if (self.iteration % self.LOG_PERIOD == 0 or
                self.iteration == self.MAX_ITER - 1):
            if self.summary_logger:
                for k, v in kwargs.items():
                    if k not in self.tb_ignored_keys:
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        assert isinstance(v, (float, int))
                        self.summary_logger.add_scalar(k, v, self.iteration)
