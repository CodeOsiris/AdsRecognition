import csv

#to_write = []
#names = []
#fr = open("dataset/names")
#i = 0
#for line in fr.readlines():
#    if i >= 3:
#        break
#    key = line.split(":")[0].split("*")[-1]
#    names.append(key)
#    i += 1
#fr.close()
#to_write.append(names)

#imputed = []
#fr = open("mydata.csv")
#for line in fr.readlines():
#    imputed.append(line.strip('\n').split(',')[-3:])
#fr.close()

table = []
fr = open("dataset/normal_distribution")
for line in fr.readlines():
    row = []
    for num in line.split():
        row.append(float(num))
    table.append(row)
fr.close()

sample_set = []
fr = open("dataset/ad.imputed")
for line in fr.readlines():
    sample = []
    line = line.split()
    for attribute in line:
        attribute = attribute.strip()
        sample.append(attribute)
    sample_set.append(sample)
#    to_write.append(sample[:3])
fr.close()

def ndp(x):
    if x > 0:
        return table[int(x * 10)][int(x * 100) % 10]
    else:
        return 1.0 - table[int(-x * 10)][int(-x * 100) % 10]

total = [0, 0, 0]
for sample in sample_set:
    for i in range(3):
        total[i] = total[i] + float(sample[i])
mean = [total[0] / len(sample_set), total[1] / len(sample_set), total[2] / len(sample_set)]
sqd = [0, 0, 0]
for sample in sample_set:
    for i in range(3):
        sample[i] = float(sample[i])
        sqd[i] = sqd[i] + (sample[i] - mean[i]) * (sample[i] - mean[i])
sqd = [sqd[0] / len(sample_set), sqd[1] / len(sample_set), sqd[2] / len(sample_set)]
for sample in sample_set:
    for i in range(3):
        sample[i] = (sample[i] - mean[i]) / sqd[i]
    print '{0:5f} {0:5f} {0:5f}'.format(ndp(sample[0]), ndp(sample[1]), ndp(sample[2]))
    #print sample[:3]
#print mean
#print sqd

#fo = open("data.csv", "wb")
#writer = csv.writer(fo, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_NONNUMERIC)
#for line in to_write:
#    writer.writerow(line)
#fo.close()
#print len(names)
#print len(to_write[1])
print len(sample_set)
#fo = open("imputed", "wb")
#for line in sample_set:
#    s = ""
#    for attr in line:
#        s += attr + ' '
#    s = s[:-1] + '\n'
#    fo.write(s)
#fo.close()
