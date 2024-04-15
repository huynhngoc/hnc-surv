import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import lifelines.datasets

sns.set_style("darkgrid")


############################ KM CURVE #######################################

df = lifelines.datasets.load_g3()

df_male = df[df["sex"] == 'Male']
df_female = df[df["sex"] == 'Female']


time_male = df_male['time']
event_male = df_male['event']

time_female = df_female['time']
event_female = df_female['event']

kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()

# Fit the data
kmf1.fit(time_male, event_male, label='Group 1')
kmf2.fit(time_female, event_female, label='Group 2')


plt.figure(figsize=(8, 8))
ax = kmf1.plot_survival_function(ci_show=False, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})
kmf2.plot_survival_function(ax=ax, ci_show=False, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})

add_at_risk_counts(kmf1, kmf2, ax=ax)

plt.title('Example KM curve')
plt.tight_layout()
ax.legend(loc='lower left', fontsize=12)
ax.set_xlabel('Time Intervals', fontsize=12)
ax.set_ylabel('Survival probability', fontsize=12)
plt.show()
plt.close()

######################### ROC "#######################################################################################

np.random.seed(123)
y_true = np.random.randint(0, 2, 100)
y_scores = np.random.uniform(0.5, 0.1, 100) + (0.1 * y_true)

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Performance')
plt.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', lw=2, label='Perfect Performance')

plt.fill_between(fpr, tpr, color='skyblue', alpha=0.2)
plt.plot([], [], color='skyblue', alpha=0.5, label='AUC - shaded area')

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc="lower right", frameon=True)
plt.show()
