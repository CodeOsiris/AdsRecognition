class Naive_Bayesian:
    def __init__(self, dataset):
        self.attr = len(dataset[0])
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

nbc = Naive_Bayesian(dataset)
print "Total Ads:", nbc.ad_n
print "Total Non-Ads:", nbc.nad_n
print "Mean Value of Ads:", nbc.ad_mean
print "Square Deviation of Ads:", nbc.nad_mean
print "Mean Value of Non-Ads:", nbc.ad_sqd
print "Square Deviation of Non-Ads:", nbc.nad_sqd
