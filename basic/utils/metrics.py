def acc_metric(output, target, thredshold=0.5):
    """
    :param output:model classification output
    :param target:ground truth label
    :param threshold:正样本的阈值
    return 返回这个batch size的准确率,total,pos,neg
    """
    total_pos = 0
    total_neg = 0
    correct_pos = 0
    correct_neg = 0
    for i in range(len(output)):
        _o = output[i]
        _t = target[i]
        pred_label = 1 if _o > thredshold else 0
        if int(_t) == 1:
            total_pos += 1
            if pred_label == 1:
                correct_pos += 1
        else:
            total_neg += 1
            if pred_label == 0:
                correct_neg += 1
    total = total_pos + total_neg
    correct_total = correct_pos + correct_neg
    return  correct_pos, total_pos,  correct_neg , total_neg

class Counter(object):
    """统计均值的类

    Attributes：
        val_current:最后一次更新的结果
        val_list:所有val的历史记录
        avg:平均值
        sum:总数
        count:len(val_list)，计数次数

    Example:
        time_counter = Counter()
        time_counter.update(time.time)
        #... some code here
        time_counter.update(time.time)
        return time_counter.interval(1)
    """

    def __init__(self):
        self.val_current = 0
        self.val_list = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.key_to_val = {}

    def reset(self):
        self.val_current = 0
        self.val_list = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def addval(self, val, n=1, key=''):
        self.val_current = val
        self.val_list.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if key != '':
            self.key_to_val[key] = val

    def interval(self, interval=1):
        """
        返回间隔的计数
        :param interval:间隔，默认返回最近两个的差
        :return:
        """
        if interval < self.count:
            return self.val_list[-1] - self.val_list[-1 - interval]
        else:
            return "error: interval out of range。"

    def key_interval(self, key_st='', key_ed=''):
        """
        以key的方式返回间隔
        :param key_st:required，返回的起点
        :param key_ed:计数终点
        :return:
        """
        if key_st == '' and key_ed == '':
            return "error: interval key miss"

        if key_ed == '':
            return self.val_list[-1] - self.key_to_val[key_st]
        else:
            return self.key_to_val[key_ed] - self.key_to_val[key_st]
