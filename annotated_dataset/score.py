


with open('corrected.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]

tp = 0.0
tn = 0.0
fp = 0.0
fn = 0.0

for line in lines:
    split = line.split(';')
    if split[0] == split[2]:
        if split[0] == '0':
            tn += 1
        else:
            tp += 1
    else:
        if split[0] == '1':
            fp += 1
        else:
            fn += 1

f1 = (2 * tp) / (2 * tp + fp + fn)

print(f1)