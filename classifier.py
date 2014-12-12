import sys
import math
import random
import time

class Naive_Bayesian:
    def __init__(self, dataset):
        #self.dataset = dataset
        self.ad_n = 0
        self.nad_n = 0
        self.ad_total = [0] * len(dataset[0])
        self.nad_total = [0] * len(dataset[0])
        for sample in dataset:
            if sample[-1] == "ad":
                self.ad_n += 1
                for i in range(len(dataset[0]) - 1):
                    self.ad_total[i] += sample[i]
            else:
                self.nad_n += 1
                for i in range(len(dataset[0]) - 1):
                    self.nad_total[i] += sample[i]
        self.ad_mean = [self.ad_total[0] / self.ad_n, self.ad_total[1] / self.ad_n, self.ad_total[2] / self.ad_n]
        self.nad_mean = [self.nad_total[0] / self.nad_n, self.nad_total[1] / self.nad_n, self.nad_total[2] / self.nad_n]
        self.ad_sqd = [0, 0, 0]
        self.nad_sqd = [0, 0, 0]
        for sample in dataset:
            if sample[-1] == "ad":
                for i in range(3):
                    self.ad_sqd[i] += (sample[i] - self.ad_mean[i]) * (sample[i] - self.ad_mean[i])
            else:
                for i in range(3):
                    self.nad_sqd[i] += (sample[i] - self.nad_mean[i]) * (sample[i] - self.nad_mean[i])
        self.ad_sqd = [self.ad_sqd[0] / self.ad_n, self.ad_sqd[1] / self.ad_n, self.ad_sqd[2] / self.ad_n]
        self.nad_sqd = [self.nad_sqd[0] / self.nad_n, self.nad_sqd[1] / self.nad_n, self.nad_sqd[2] / self.nad_n]

    def ndp(self, x, mean, std):
        x = (x - mean) / math.sqrt(std)
        return math.exp(-x * x / 2) / math.sqrt(math.acos(-1) * 2)

    def probability(self, sample):
        total = self.ad_n + self.nad_n
        ad_p = float(self.ad_n) / total
        nad_p = float(self.nad_n) / total
        for i in range(3):
            ad_p *= self.ndp(sample[i], self.ad_mean[i], self.ad_sqd[i])
            nad_p *= self.ndp(sample[i], self.nad_mean[i], self.nad_sqd[i])
        for i in range(3, len(sample)):
            laplace_ad = float(self.ad_total[i])
            laplace_nad = float(self.nad_total[i])
            if laplace_ad == 0:
                laplace_ad += 1
            if laplace_nad == 0:
                laplace_nad += 1
            if sample[i] == 1:
                ad_p *= float(laplace_ad) / total
                nad_p *= float(laplace_nad) / total
            else:
                ad_p *= 1 - float(laplace_ad) / total
                nad_p *= 1 - float(laplace_nad) / total
        category = ""
        if ad_p > nad_p:
            category = "ad"
        else:
            category = "nonad"
        return (ad_p, nad_p, category)

    def increment(self, sample):
        new_mean = [0, 0, 0]
        new_sqd = [0, 0, 0]
        if sample[-1] == "ad":
            for i in range(3):
                new_mean[i] = (self.ad_mean[i] * self.ad_n + sample[i]) / (self.ad_n + 1)
                new_sqd[i] = 2 * (self.ad_mean[i] - new_mean[i]) * self.ad_total[i] + self.ad_n * (new_mean[i] * new_mean[i] - self.ad_mean[i] * self.ad_mean[i]) + (sample[i] - new_mean[i]) * (sample[i] - new_mean[i])
                self.ad_mean[i] = new_mean[i]
                self.ad_sqd[i] = (self.ad_sqd[i] * self.ad_n + new_sqd[i]) / (self.ad_n + 1)
            for i in range(len(sample) - 1):
                self.ad_total[i] += sample[i]
            self.ad_n += 1
        else:
            for i in range(3):
                new_mean[i] = (self.nad_mean[i] * self.nad_n + sample[i]) / (self.nad_n + 1)
                new_sqd[i] = 2 * (self.nad_mean[i] - new_mean[i]) * self.nad_total[i] + self.nad_n * (new_mean[i] * new_mean[i] - self.nad_mean[i] * self.nad_mean[i]) + (sample[i] - new_mean[i]) * (sample[i] - new_mean[i])
                self.nad_mean[i] = new_mean[i]
                self.nad_sqd[i] = (self.nad_sqd[i] * self.nad_n + new_sqd[i]) / (self.nad_n + 1)
            for i in range(len(sample) - 1):
                self.nad_total[i] += sample[i]
            self.nad_n += 1

def read():
    fr = open("dataset/ad.imputed")
    dataset = []
    for sample in fr.readlines():
        sample = sample.split()
        for i in range(len(sample) - 1):
            if i < 3:
                sample[i] = float(sample[i])
            else:
                sample[i] = int(sample[i])
        dataset.append(sample)
    fr.close()
    return dataset

def mask(dataset, mask_str):
    subset = []
    for i in range(len(dataset)):
        subset.append(dataset[i][:3])
    for i in range(3, len(mask_str)):
        if mask_str[i] == '1':
            for j in range(len(dataset)):
                subset[j].append(dataset[j][i])
    return subset

def ten_fold(dataset):
    folds = []
    index = []
    for i in range(10):
        folds.append([])
        index.append([])
    for i in range(len(dataset)):
        r = random.random()
        folds[int(r * 10)].append(dataset[i])
        index[int(r * 10)].append(i)
    total = 0
    return (folds, index)

def reduce_weight(weight):
    return weight * math.exp(-0.9)

def two_stage(dataset, threshold):
    (folds, index) = ten_fold(dataset)
    correct = 0
    total = 0
    u = set()
    undetermined = []
    for i in range(10):
        train = []
        test = folds[i]
        test_index = index[i]
        for j in range(10):
            if i != j:
                train.extend(folds[j])
        nbc = Naive_Bayesian(train)
        for i in range(len(test)):
            (ad_p, nad_p, label) = nbc.probability(test[i][:-1])
            if ad_p / nad_p >= threshold:
                u.add(test_index[i])
    for num in u:
        undetermined.append(dataset[num])
    nbc_a = Naive_Bayesian(dataset)
    nbc_b = Naive_Bayesian(undetermined)
    return (nbc_a, nbc_b)

def cross_validation(dataset):
    (folds, index) = ten_fold(dataset)
    correct = 0
    total = 0
    false_pos = 0
    neg = 0
    true_pos = 0
    pos = 0
    for i in range(10):
        train = []
        test = folds[i]
        testset = folds[i]
        for j in range(10):
            if i != j:
                train.extend(folds[j])
        nbc = Naive_Bayesian(train)
        fold_correct = 0
        fold_total = len(testset)
        fold_false_pos = 0
        fold_neg = 0
        fold_true_pos = 0
        fold_pos = 0
        for test in testset:
            (ad_p, nad_p, label) = nbc.probability(test[:-1])
            nbc.increment(test)
            if label == test[-1]:
                fold_correct += 1
            if test[-1] == "nonad":
                fold_neg += 1
                if label == "ad":
                    fold_false_pos += 1
            else:
                fold_pos += 1
                if label == "ad":
                    fold_true_pos += 1
        false_pos += fold_false_pos
        neg += fold_neg
        true_pos += fold_true_pos
        pos += fold_pos
        correct += fold_correct
        total += fold_total
    s = float(pos) / total
    p = float(true_pos + false_pos) / total
    mcc = (float(true_pos) / total - s * p) / math.sqrt(s * p * (1 - s) * (1 - p))
    print "Cross Validation Accuracy: {0:.2%}".format(float(correct) / total)
    print "Cross Validation TPR: {0:.2%}".format(float(true_pos) / pos)
    print "Cross Validation FPR: {0:.2%}".format(float(false_pos) / neg)
    print "Cross Validation MCC:", mcc
    #return float(correct) / total
    return (correct, total, true_pos, pos, false_pos, neg)

def cross_validation_stage(dataset, threshold):
    (folds, index) = ten_fold(dataset)
    correct = 0
    total = 0
    false_pos = 0
    neg = 0
    true_pos = 0
    pos = 0
    for i in range(10):
        train = []
        testset = folds[i]
        for j in range(10):
            if i != j:
                train.extend(folds[j])
        (nbc_a, nbc_b) = two_stage(train, threshold)
        fold_correct = 0
        fold_total = len(testset)
        fold_false_pos = 0
        fold_neg = 0
        fold_true_pos = 0
        fold_pos = 0
        for test in testset:
            (ad_p, nad_p, label) = nbc_a.probability(test[:-1])
            nbc_a.increment(test)
            if nad_p == 0 or ad_p / nad_p >= threshold:
                (ad_p, nad_p, label) = nbc_b.probability(test[:-1])
                nbc_b.increment(test)
            if label == test[-1]:
                fold_correct += 1
            if test[-1] == "nonad":
                fold_neg += 1
                if label == "ad":
                    fold_false_pos += 1
            else:
                fold_pos += 1
                if label == "ad":
                    fold_true_pos += 1
        false_pos += fold_false_pos
        neg += fold_neg
        true_pos += fold_true_pos
        pos += fold_pos
        correct += fold_correct
        total += fold_total
    s = float(pos) / total
    p = float(true_pos + false_pos) / total
    mcc = (float(true_pos) / total - s * p) / math.sqrt(s * p * (1 - s) * (1 - p))
    print "Cross Validation Accuracy: {0:.2%}".format(float(correct) / total)
    print "Cross Validation TPR: {0:.2%}".format(float(true_pos) / pos)
    print "Cross Validation FPR: {0:.2%}".format(float(false_pos) / neg)
    print "Cross Validation MCC:", mcc
    #return float(correct) / total
    return (correct, total, true_pos, pos, false_pos, neg)

def cross_validation_weighted(dataset, threshold, masks):
    (folds, index) = ten_fold(dataset)
    correct = 0
    total = 0
    false_pos = 0
    neg = 0
    true_pos = 0
    pos = 0
    for i in range(10):
        train = []
        testset = folds[i]
        for j in range(10):
            if i != j:
                train.extend(folds[j])
        nbc_a, nbc_b, subset = [], [], []
        for j in range(len(masks)):
            subset.append(mask(dataset, masks[j]))
            (nbc_as, nbc_bs) = two_stage(subset[j], threshold)
            nbc_a.append(nbc_as)
            nbc_b.append(nbc_bs)
        fold_correct = 0
        fold_total = len(testset)
        fold_false_pos = 0
        fold_neg = 0
        fold_true_pos = 0
        fold_pos = 0
        weight = [1.0] * len(masks)
        for test in testset:
            labels = [""] * len(masks)
            weighted = 0
            weighted_label = ""
            for k in range(len(masks)):
                subtest = mask([test], masks[k])[0]
                (ad_p, nad_p, label) = nbc_a[k].probability(subtest[:-1])
                if ad_p / nad_p >= threshold:
                    (ad_p, nad_p, label) = nbc_b[k].probability(subtest[:-1])
                labels[k] = label
                if label == "ad":
                    weighted += weight[k]
            if weighted < sum(weight) / 2:
                weighted_label = "nonad"
            else:
                weighted_label = "ad"
            if weighted_label != subtest[-1]:
                for k in range(len(masks)):
                    subtest = mask([test], masks[k])[0]
                    if labels[k] != test[-1]:
                        weight[k] = reduce_weight(weight[k])
                        nbc_a[k].increment(subtest)
                        nbc_b[k].increment(subtest)
            if weighted_label == test[-1]:
                fold_correct += 1
            if test[-1] == "nonad":
                fold_neg += 1
                if weighted_label == "ad":
                    fold_false_pos += 1
            else:
                fold_pos += 1
                if weighted_label == "ad":
                    fold_true_pos += 1
        #print "Fold Accuracy: {0:.2%}".format(float(fold_correct) / fold_total)
        false_pos += fold_false_pos
        neg += fold_neg
        true_pos += fold_true_pos
        pos += fold_pos
        correct += fold_correct
        total += fold_total
    s = float(pos) / total
    p = float(true_pos + false_pos) / total
    mcc = (float(true_pos) / total - s * p) / math.sqrt(s * p * (1 - s) * (1 - p))
    print "Cross Validation Accuracy: {0:.2%}".format(float(correct) / total)
    print "Cross Validation TPR: {0:.2%}".format(float(true_pos) / pos)
    print "Cross Validation FPR: {0:.2%}".format(float(false_pos) / neg)
    print "Cross Validation MCC:", mcc
    #return float(correct) / total
    #return mcc
    return (correct, total, true_pos, pos, false_pos, neg)

def statistic(iters):
    fo = open("statistic", "wb")

    correct = 0
    total = 0
    true_pos = 0
    pos = 0
    false_neg = 0
    neg = 0
    for i in range(iters):
        print "Training Round %d:" % (i + 1)
        prev = time.time()
        (r_correct, r_total, r_true_pos, r_pos, r_false_neg, r_neg) = cross_validation_weighted(dataset, 1, masks)
        cur = time.time()
        print "Time Used:", cur - prev
        correct += r_correct
        total += r_total
        true_pos += r_true_pos
        pos += r_pos
        false_neg += r_false_neg
        neg += r_neg
    s = float(pos) / total
    p = float(true_pos + false_neg) / total
    mcc = (float(true_pos) / total - s * p) / math.sqrt(s * p * (1 - s) * (1 - p))
    print "Exponential Weighted Average:"
    print "Average Accuracy: {0:.2%}".format(float(correct) / total)
    print "Average TPR: {0:.2%}".format(float(true_pos) / pos)
    print "Average FPR: {0:.2%}".format(float(false_neg) / neg)
    print "Average MCC:", mcc
    fo.write("Exponential Weighted Average:\n")
    fo.write("Average Accuracy: {0:.2%}\n".format(float(correct) / total))
    fo.write("Average TPR: {0:.2%}\n".format(float(true_pos) / pos))
    fo.write("Average FPR: {0:.2%}\n".format(float(false_neg) / neg))
    fo.write("Average MCC:%f\n\n" % mcc)

    correct = 0
    total = 0
    true_pos = 0
    pos = 0
    false_neg = 0
    neg = 0
    for i in range(iters):
        print "Training Round %d:" % (i + 1)
        prev = time.time()
        (r_correct, r_total, r_true_pos, r_pos, r_false_neg, r_neg) = cross_validation_stage(dataset, 1)
        cur = time.time()
        print "Time Used:", cur - prev
        correct += r_correct
        total += r_total
        true_pos += r_true_pos
        pos += r_pos
        false_neg += r_false_neg
        neg += r_neg
    s = float(pos) / total
    p = float(true_pos + false_neg) / total
    mcc = (float(true_pos) / total - s * p) / math.sqrt(s * p * (1 - s) * (1 - p))
    print "Two Stage:"
    print "Average Accuracy: {0:.2%}".format(float(correct) / total)
    print "Average TPR: {0:.2%}".format(float(true_pos) / pos)
    print "Average FPR: {0:.2%}".format(float(false_neg) / neg)
    print "Average MCC:", mcc
    fo.write("Two Stage:\n")
    fo.write("Average Accuracy: {0:.2%}\n".format(float(correct) / total))
    fo.write("Average TPR: {0:.2%}\n".format(float(true_pos) / pos))
    fo.write("Average FPR: {0:.2%}\n".format(float(false_neg) / neg))
    fo.write("Average MCC:%f\n\n" % mcc)

    correct = 0
    total = 0
    true_pos = 0
    pos = 0
    false_neg = 0
    neg = 0
    for i in range(iters):
        print "Training Round %d:" % (i + 1)
        prev = time.time()
        (r_correct, r_total, r_true_pos, r_pos, r_false_neg, r_neg) = cross_validation(dataset)
        cur = time.time()
        print "Time Used:", cur - prev
        correct += r_correct
        total += r_total
        true_pos += r_true_pos
        pos += r_pos
        false_neg += r_false_neg
        neg += r_neg
    s = float(pos) / total
    p = float(true_pos + false_neg) / total
    mcc = (float(true_pos) / total - s * p) / math.sqrt(s * p * (1 - s) * (1 - p))
    print "Naive Bayesian:"
    print "Average Accuracy: {0:.2%}".format(float(correct) / total)
    print "Average TPR: {0:.2%}".format(float(true_pos) / pos)
    print "Average FPR: {0:.2%}".format(float(false_neg) / neg)
    print "Average MCC:", mcc
    fo.write("Naive Bayesian:\n")
    fo.write("Average Accuracy: {0:.2%}\n".format(float(correct) / total))
    fo.write("Average TPR: {0:.2%}\n".format(float(true_pos) / pos))
    fo.write("Average FPR: {0:.2%}\n".format(float(false_neg) / neg))
    fo.write("Average MCC:%f\n\n" % mcc)
    fo.close()

def main():
    dataset = read()
    fr = open("mask")
    masks = fr.readlines()
    fr.close()
    for i in range(len(masks)):
        masks[i] = masks[i].split()
    args = sys.argv
    if len(args) > 1:
        iters = int(args[1])
        statistic(iters)

if __name__ == "__main__":
    main()
