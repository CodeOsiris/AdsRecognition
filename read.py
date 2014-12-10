import csv
import math

#to_write = []
#names = []
#fr = open("dataset/names")
#i = 0
#for line in fr.readlines():
#    if i >= 4:
#        break
#    key = line.split(":")[0].split("*")[-1]
#    names.append(key)
#    i += 1
#fr.close()
#to_write.append(names)

imputed = []
fr = open("mydata.csv")
for line in fr.readlines():
    imputed.append(line.strip('\n').split(',')[-4:])
fr.close()

table = []
fr = open("dataset/normal_distribution")
for line in fr.readlines():
    row = []
    for num in line.split():
        row.append(float(num))
    table.append(row)
fr.close()

sample_set = []
cnt = 0
##fr = open("dataset/ad.imputed")
fr = open("dataset/ad.data")
for line in fr.readlines():
##    sample = []
##    line = line.split()
##    for attribute in line:
##        attribute = attribute.strip()
##        sample.append(attribute)
    sample = line.split(',')
    for i in range(len(sample)):
        if i < 4:
            sample[i] = imputed[cnt][i]
        else:
            sample[i] = sample[i].strip()
    sample[-1] = sample[-1][:-1]
    sample_set.append(sample)
    cnt += 1
#    to_write.append(sample[:4])
#fr.close()

#def ndp(x):
#    pi = math.acos(-1)
#    return 1.0/math.sqrt(pi * 2) * math.exp(-x * x / 2)
#
#ad_total = [0, 0, 0]
#nad_total = [0, 0, 0]
#ad_n = 0
#nad_n = 0
#for sample in sample_set:
#    if sample[-1] == "ad":
#        for i in range(3):
#            ad_total[i] += float(sample[i])
#        ad_n += 1
#    else:
#        for i in range(3):
#            nad_total[i] += float(sample[i])
#        nad_n += 1
#ad_mean = [ad_total[0] / ad_n, ad_total[1] / ad_n, ad_total[2] / ad_n]
#nad_mean = [nad_total[0] / nad_n, nad_total[1] / nad_n, nad_total[2] / nad_n]
#ad_std = [0, 0, 0]
#nad_std = [0, 0, 0]
#for sample in sample_set:
#    for i in range(3):
#        sample[i] = float(sample[i])
#        if sample[-1] == "ad":
#            ad_std[i] = ad_std[i] + (sample[i] - ad_mean[i]) * (sample[i] - ad_mean[i])
#        else:
#            nad_std[i] = nad_std[i] + (sample[i] - nad_mean[i]) * (sample[i] - nad_mean[i])
#ad_std = map(math.sqrt, [ad_std[0] / ad_n, ad_std[1] / ad_n, ad_std[2] / ad_n])
#nad_std = map(math.sqrt, [nad_std[0] / nad_n, nad_std[1] / nad_n, nad_std[2] / nad_n])
#for sample in sample_set:
#    for i in range(3):
#        if sample[-1] == "ad":
#            sample[i] = (sample[i] - ad_mean[i]) / ad_std[i]
#        else:
#            sample[i] = (sample[i] - nad_mean[i]) / nad_std[i]
#    print "%10.7f %10.7f %10.7f" % (ndp(sample[0]), ndp(sample[1]), ndp(sample[2]))
#    print sample[:4], sample[-1]
#print mean
#print std

#fo = open("data.csv", "wb")
#writer = csv.writer(fo, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_NONNUMERIC)
#for line in to_write:
#    writer.writerow(line)
#fo.close()

#print len(names)
#print len(to_write[1])
#print len(sample_set)
#print ad_mean, ad_std
#print nad_mean, nad_std
#print ad_n, nad_n

fo = open("dataset/ad.imputed", "wb")
for line in sample_set:
    s = ""
    for attr in line:
        s += attr + ' '
    s = s[:-1] + '\n'
    fo.write(s)
fo.close()
