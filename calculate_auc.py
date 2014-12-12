import operator

fr = open("statistics/roc_ewa")
points = fr.readlines()
fr.close()
area = 0
for i in range(len(points)):
    points[i] = map(float, points[i].split(','))
points = sorted(points, key = operator.itemgetter(1, 0))
prev = points[0]
for i in range(1, len(points)):
    cur = points[i]
    area += (cur[1] - prev[1]) * (cur[0] + prev[0]) / 2
    prev = cur
print "Exponential Weighted Average AUC:\n%f" % area

fr = open("statistics/roc_ts")
points = fr.readlines()
fr.close()
area = 0
for i in range(len(points)):
    points[i] = map(float, points[i].split(','))
points = sorted(points, key = operator.itemgetter(1, 0))
prev = points[0]
for i in range(1, len(points)):
    cur = points[i]
    area += (cur[1] - prev[1]) * (cur[0] + prev[0]) / 2
    prev = cur
print "Two Stage AUC:\n%f" % area

fr = open("statistics/roc_nb")
points = fr.readlines()
fr.close()
area = 0
for i in range(len(points)):
    points[i] = map(float, points[i].split(','))
points = sorted(points, key = operator.itemgetter(1, 0))
prev = points[0]
for i in range(1, len(points)):
    cur = points[i]
    area += (cur[1] - prev[1]) * (cur[0] + prev[0]) / 2
    prev = cur
print "Naive Bayesian AUC:\n%f" % area

fr = open("statistics/roc_tsff")
points = fr.readlines()
fr.close()
area = 0
for i in range(len(points)):
    points[i] = map(float, points[i].split(','))
points = sorted(points, key = operator.itemgetter(1, 0))
prev = points[0]
for i in range(1, len(points)):
    cur = points[i]
    area += (cur[1] - prev[1]) * (cur[0] + prev[0]) / 2
    prev = cur
print "Two Stage After Feature Filter AUC:\n%f" % area
