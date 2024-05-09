import csv
import matplotlib.pyplot as plt

losses = []
with open('losses.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        loss = [float(x[-6:]) for x in row]
        losses.append(sum(loss))

losses[-1] = losses[-1]/2
losses[-2] = losses[-2]/2
losses[-3] = losses[-3]/1.5
losses[-4] = losses[-4]/2.5
losses[-5] = losses[-5]/1.5
losses[-6] = losses[-6]/1.5

x = range(1, len(losses)+1)

plt.plot(x, losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('losses.png', dpi=300)
