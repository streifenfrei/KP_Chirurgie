import numpy as np
import matplotlib.pyplot as plt  


x = [10, 20, 30, 40, 50]
y_csl_run1 = [0.9491617090319091, 0.9767441860465116, 0.9859383450513791, 0.9951325040562466, 1.0]
y_nseg_run1 = [0.9494623655913978, 0.975268817204301, 0.9870967741935484, 0.9956989247311828, 1.0]

y_csl_run2 = [0.9520697167755992, 0.9754901960784313, 0.9874727668845316, 0.9940087145969498, 1.0]
y_csl_run3 = [0.9365979381443299, 0.9747422680412371, 0.9860824742268042, 0.9953608247422681, 1.0]


y_csl_max = [max((y_csl_run1[i], y_csl_run2[i], y_csl_run3[i])) for i in range(len(x))]
y_csl_min = [min((y_csl_run1[i], y_csl_run2[i], y_csl_run3[i])) for i in range(len(x))]

line1, = plt.plot(x, y_csl_run1, color='blue', lw=1, ls='-', marker='o', ms=4, label='csl with seg')
line2, = plt.plot(x, y_nseg_run1, color='red', lw=1,  marker='^', ms=4, label='csl without seg')

plt.fill_between(x, y_csl_min, y_csl_max, color=(191/256, 191/256, 255/256), alpha=0.7)


plt.title('distance threshold score')
plt.xlabel('distance threshold (in pixel)')
plt.ylabel('percent of matching landmarks')
plt.legend()
plt.show()
