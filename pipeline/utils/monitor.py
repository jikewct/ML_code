import time


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(
        self,
        initVal=None,
        momentum=0.99,
    ):
        self.momentum = momentum
        self.initVal = initVal
        self.val = initVal
        self.avg = 0

    def reset(self):
        self.val = self.initVal
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

    def inc(self, val=1):
        self.update(self.val + val)


class TimeMeter(object):

    def __init__(self):
        self.reset()

    def start(self):
        self.preTime = time.time()
        self.currentTime = self.preTime
        return self

    def reset(self):
        self.startTime = time.time()
        self.currentTime = 0
        self.preTime = 0

    def stop(self):
        self.currentTime = time.time()
        return self

    def interval(self):
        return self.currentTime - self.preTime

    def intervalToNow(self):
        return time.time() - self.preTime

    def totalTime(self):
        self.currentTime = time.time()
        return self.currentTime - self.startTime


class HistogramMeter(object):

    def __init__(self, desc):
        self.count_dict = dict()
        self.sum_dict = dict()
        self.desc = desc

    def add_metric_with_group(self, metrics, groups):
        for i in range(groups.shape[0]):
            key = f"{self.desc}-{groups[i].item()}"
            if key in self.sum_dict:
                self.sum_dict[key] += metrics[i].item()
                self.count_dict["cnt-" + key] += 1
            else:
                self.sum_dict[key] = metrics[i].item()
                self.count_dict["cnt-" + key] = 1

    def get_avg_metric_dict(self):
        for key in self.sum_dict.keys():
            self.sum_dict[key] = self.sum_dict[key] / self.count_dict["cnt-" + key]

        self.sum_dict.update(self.count_dict)
        return self.sum_dict

    def get_avg_metric_dict_without_cnt(self):
        for key in self.sum_dict.keys():
            self.sum_dict[key] = self.sum_dict[key] / self.count_dict["cnt-" + key]
        return self.sum_dict

    def reset(self):
        self.count_dict.clear()
        self.sum_dict.clear()
