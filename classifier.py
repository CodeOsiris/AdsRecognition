import math

class Naive_Bayesian:
    def __init__(self, dataset):
        self.attr_n = len(dataset[0])
        self.dataset = dataset
        self.ad_n = 0
        self.nad_n = 0
        self.ad_total = [0] * len(dataset[0])
        self.nad_total = [0] * len(dataset[0])
        for x in dataset:
            if x[-1] == "ad":
                self.ad_n += 1
                for i in range(len(dataset[0]) - 1):
                    self.ad_total[i] += x[i]
            else:
                self.nad_n += 1
                for i in range(len(dataset[0]) - 1):
                    self.nad_total[i] += x[i]
        self.ad_mean = [self.ad_total[0] / self.ad_n, self.ad_total[1] / self.ad_n, self.ad_total[2] / self.ad_n, self.ad_total[3] / self.ad_n]
        self.nad_mean = [self.nad_total[0] / self.nad_n, self.nad_total[1] / self.nad_n, self.nad_total[2] / self.nad_n, self.nad_total[3] / self.nad_n]
        self.ad_sqd = [0, 0, 0, 0]
        self.nad_sqd = [0, 0, 0, 0]
        for x in dataset:
            if x[-1] == "ad":
                for i in range(4):
                    self.ad_sqd[i] += (x[i] - self.ad_mean[i]) * (x[i] - self.ad_mean[i])
            else:
                for i in range(4):
                    self.nad_sqd[i] += (x[i] - self.nad_mean[i]) * (x[i] - self.nad_mean[i])
        self.ad_sqd = [self.ad_sqd[0] / self.ad_n, self.ad_sqd[1] / self.ad_n, self.ad_sqd[2] / self.ad_n, self.ad_sqd[3] / self.ad_n]
        self.nad_sqd = [self.nad_sqd[0] / self.nad_n, self.nad_sqd[1] / self.nad_n, self.nad_sqd[2] / self.nad_n, self.nad_sqd[3] / self.nad_n]

    def ndp(self, x, mean, std):
        x = (x - mean) / math.sqrt(std)
        return math.exp(-x * x / 2) / math.sqrt(math.acos(-1) * 2)

    def probability(self, input_v):
        total = self.ad_n + self.nad_n
        ad_p = float(self.ad_n) / total
        nad_p = float(self.nad_n) / total
        for i in range(3):
            ad_p *= self.ndp(input_v[i], self.ad_mean[i], self.ad_sqd[i])
            nad_p *= self.ndp(input_v[i], self.nad_mean[i], self.nad_sqd[i])
        for i in range(3, len(input_v)):
            if input_v[i] == 1:
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

    def accuracy(self):
        correct = 0
        total = self.ad_n + self.nad_n
        for v in self.dataset:
            result = self.probability(v[:-1])
            if result[2] == v[-1]:
                correct += 1
        return float(correct) / total

def ten_fold(dataset):
    folds = []
    return folds

def read():
    fr = open("dataset/ad.imputed")
    dataset = []
    for sample in fr.readlines():
        sample = sample.split()
        for i in range(len(sample) - 1):
            if i < 4:
                sample[i] = float(sample[i])
            else:
                sample[i] = int(sample[i])
        dataset.append(sample)
    fr.close()
    return dataset

nbc = Naive_Bayesian(read())

#print "Total Ads:", nbc.ad_n
#print "Total Non-Ads:", nbc.nad_n
#print "Mean Value of Ads:", nbc.ad_mean
#print "Square Deviation of Ads:", nbc.nad_mean
#print "Mean Value of Non-Ads:", nbc.ad_sqd
#print "Square Deviation of Non-Ads:", nbc.nad_sqd
print "Accuracy: {0:.2%}".format(nbc.accuracy())
