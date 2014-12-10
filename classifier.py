import math
import random

class Naive_Bayesian:
    def __init__(self, dataset):
        self.attr_n = len(dataset[0])
        self.dataset = dataset
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
            if sample[i] == 1:
                ad_p *= float(self.ad_total[i]) / total
                nad_p *= float(self.nad_total[i]) / total
            else:
                ad_p *= 1 - float(self.ad_total[i]) / total
                nad_p *= 1 - float(self.nad_total[i]) / total
        category = ""
        if ad_p > nad_p:
            category = "ad"
        else:
            category = "nonad"
        return (ad_p, nad_p, category)

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

def ten_fold(dataset):
    folds = []
    for i in range(10):
        train = []
        test = []
        for sample in dataset:
            if random.random() < 0.9:
                train.append(sample)
            else:
                test.append(sample)
        folds.append((train, test))
    return folds

def accuracy(nbc, testset):
    correct = 0
    total = len(testset)
    for test in testset:
        (ad_p, nad_p, label) = nbc.probability(test[:-1])
        if label == test[-1]:
            correct += 1
    return (correct, total)

def cross_validation(dataset):
    folds = ten_fold(dataset)
    correct = 0
    total = 0
    for (train, test) in folds:
        nbc = Naive_Bayesian(train)
        (fold_correct, fold_total) = accuracy(nbc, test)
        #print "Accuracy: {0:.2%}".format(float(fold_correct) / fold_total)
        correct += fold_correct
        total += fold_total
    print "Cross Validation Accuracy: {0:.2%}".format(float(correct) / total)
    return float(correct) / total

def main():
    dataset = read()
    #cross_validation(dataset)

if __name__ == "__main__":
    main()
