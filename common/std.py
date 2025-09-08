import numpy as np

x = [8.409,
8.2,
7.368,
6.133,
9.146,
5.325,
6.264,
4.747,
8.115,
6.188

]
total_mean = 6.989
res = 0
for i in x:
    res += np.square(i-total_mean)

f = (res + 29.253)/10
print(f)
