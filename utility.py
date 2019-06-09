import numpy as np

def parse_csv(filename):
    f = open(filename)
    linesr = f.readlines()
    f.close()
    total = 0
    first = True
    lines = []
    labels = []
    for line in linesr:
        if first:
            first = False
            continue
        line = line.strip().split(',')
        line = map(lambda x: int(x), line)
        labels.append(line[0])
        lines.append(line[1:])
        total += 1
    return labels

y_train = parse_csv('sign_mnist_train.csv')

count = {}
for y in y_train:
    if count.get(y) is None:
        count[y] = 0
    count[y] = count[y] + 1

values = np.array(count.values())
print np.mean(values)
print np.std(values)
print np.max(values)
print np.min(values)