step = 150

fr = open("statistics/roc_ewa")
points = fr.readlines()
fr.close()
fo = open("statistics/roc_ewa_extract.csv", "wb")
for i in range(len(points)):
    if i % step == 0:
        fo.write(points[i])
fo.close()

fr = open("statistics/roc_ts")
points = fr.readlines()
fr.close()
fo = open("statistics/roc_ts_extract.csv", "wb")
for i in range(len(points)):
    if i % step == 0:
        fo.write(points[i])
fo.close()

fr = open("statistics/roc_nb")
points = fr.readlines()
fr.close()
fo = open("statistics/roc_nb_extract.csv", "wb")
for i in range(len(points)):
    if i % step == 0:
        fo.write(points[i])
fo.close()

fr = open("statistics/roc_tsff")
points = fr.readlines()
fr.close()
fo = open("statistics/roc_tsff_extract.csv", "wb")
for i in range(len(points)):
    if i % step == 0:
        fo.write(points[i])
fo.close()
