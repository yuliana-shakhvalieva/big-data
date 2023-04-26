import numpy as np

with open('AB_NYC_2019.csv') as f:
    data = np.loadtxt(f)

print(np.mean(data))
print(np.var(data))
