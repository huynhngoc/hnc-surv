import h5py
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts, plot_interval_censored_lifetimes
from lifelines.statistics import pairwise_logrank_test
import numpy as np


ds_file = '../datasets/outcome_ous.h5'
test_file = '../perf/OS_CT_PET_tumor_loglikelihood_concat/test/prediction_test.h5'

with h5py.File(ds_file, 'r') as f:
    pids = np.concatenate([f[f'fold_{i}']['patient_idx'][:] for i in range(5)])
    clinical_data = np.concatenate([f[f'fold_{i}']['clinical'][:] for i in range(5)])

info_df = pd.DataFrame({'pid': pids, 'stage_iii_iv': clinical_data[..., -1]})

with h5py.File(test_file, 'r') as f:
    pids = f['patient_idx'][:]
    predicted = f['predicted'][:]
    y = f['y'][:]

real_time = y[:, -1]
event = y[:, -2]
est_time = np.sum(y[:, :10], axis=-1)
predicted_time = np.cumprod(predicted, axis=-1)
df = pd.DataFrame({
    'pid': pids,
    'event': event,
    'real_time': real_time,
    'est_time': est_time,
    'predicted_time': np.sum(np.cumprod(predicted, axis=-1) > 0.5, axis=-1),
    **{f'predicted_{i}': np.cumprod(predicted, axis=-1)[:, i] for i in range(10)}
})


all_df = info_df.merge(df, how='left', on='pid')

covariate0 = all_df[all_df['stage_iii_iv'] == 0]
covariate1 = all_df[all_df['stage_iii_iv'] == 1]

## original data
plt.figure(figsize=(8, 8))

kmf1 = KaplanMeierFitter()
kmf1.fit(durations=covariate0['est_time'], event_observed=covariate0['event'], label='stage I-II')

kmf2 = KaplanMeierFitter()
kmf2.fit(durations=covariate1['est_time'], event_observed=covariate1['event'], label='stage III-IV')

ax = kmf1.plot_survival_function(ci_show=True, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})
kmf2.plot_survival_function(ax=ax, ci_show=True, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})

add_at_risk_counts(kmf1, kmf2, ax=ax)

plt.title('Overall Survival by Overall Stage of Disease')
plt.tight_layout()
ax.legend(loc='lower left', fontsize=12)
ax.set_xlabel('Time Intervals', fontsize=12)
ax.set_ylabel('Survival probability', fontsize=12)
plt.savefig('../KM_curev_original.png', dpi=300)

result_mlrt = pairwise_logrank_test(all_df['est_time'],
                                    all_df['stage_iii_iv'],
                                    all_df['event'])

result_mlrt.print_summary()


## predicted data
plt.figure(figsize=(8, 8))

kmf1 = KaplanMeierFitter()
# kmf1.fit(durations=covariate0['predicted_time'], event_observed=covariate0['event'], label='stage I-II')
kmf1.fit(durations=covariate0['predicted_time'], event_observed=np.ones(covariate0.shape[0]), label='stage I-II')

kmf2 = KaplanMeierFitter()
# kmf2.fit(durations=covariate1['predicted_time'], event_observed=covariate1['event'], label='stage III-IV')
kmf2.fit(durations=covariate1['predicted_time'], event_observed=np.ones(covariate1.shape[0]), label='stage III-IV')

ax = kmf1.plot_survival_function(ci_show=True, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})
kmf2.plot_survival_function(ax=ax, ci_show=True, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})

add_at_risk_counts(kmf1, kmf2, ax=ax)

plt.title('Overall Survival by Overall Stage of Disease PET/CT + GTVp model')
plt.tight_layout()
ax.legend(loc='lower left', fontsize=12)
ax.set_xlabel('Time Intervals', fontsize=12)
ax.set_ylabel('Survival probability', fontsize=12)
plt.savefig('../KM_curve_CT_PET_tumor_v2.png', dpi=300)

result_mlrt = pairwise_logrank_test(all_df['predicted_time'],
                                    all_df['stage_iii_iv'],
                                    # all_df['event'])
                                    np.ones(139))

result_mlrt.print_summary()


## predicted data (assuming an event happened for all patients died within 5 years)
plt.figure(figsize=(8, 8))

kmf1 = KaplanMeierFitter()
# kmf1.fit(durations=covariate0['predicted_time'], event_observed=covariate0['event'], label='stage I-II')
# kmf1.fit(durations=covariate0['predicted_time'], event_observed=np.ones(covariate0.shape[0]), label='stage I-II')
kmf1.fit(durations=covariate0['predicted_time'], event_observed=(covariate0['predicted_time'] < 10).astype(float), label='stage I-II')

kmf2 = KaplanMeierFitter()
# kmf2.fit(durations=covariate1['predicted_time'], event_observed=covariate1['event'], label='stage III-IV')
# kmf2.fit(durations=covariate1['predicted_time'], event_observed=np.ones(covariate1.shape[0]), label='stage III-IV')
kmf2.fit(durations=covariate1['predicted_time'], event_observed=(covariate1['predicted_time'] < 10).astype(float), label='stage III-IV')

ax = kmf1.plot_survival_function(ci_show=True, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})
kmf2.plot_survival_function(ax=ax, ci_show=True, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})

add_at_risk_counts(kmf1, kmf2, ax=ax)

plt.title('Overall Survival by Overall Stage of Disease PET/CT + GTVp model')
plt.tight_layout()
ax.legend(loc='lower left', fontsize=12)
ax.set_xlabel('Time Intervals', fontsize=12)
ax.set_ylabel('Survival probability', fontsize=12)
plt.savefig('../KM_curve_CT_PET_tumor_v3.png', dpi=300)

result_mlrt = pairwise_logrank_test(all_df['predicted_time'],
                                    all_df['stage_iii_iv'],
                                    (all_df['predicted_time'] < 10).astype(float))

result_mlrt.print_summary()

