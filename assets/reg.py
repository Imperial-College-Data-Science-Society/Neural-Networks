# scientific computing library
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['figure.figsize'] = [16.0, 12.0]
sns.set_palette(sns.color_palette("muted"))
sns.set_style("ticks")

N = 50

x = np.linspace(-1, 4, N)

y = np.sin(x) + np.random.normal(0, 0.2, N)

sns.regplot(x, y, order=2, ci=None)

plt.xlabel('$x$')
plt.ylabel('$y$')

plt.savefig('assets/Regression.eps', format='eps',
            dpi=1000, transparent=True)
