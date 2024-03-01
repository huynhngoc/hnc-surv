import h5py
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import pairwise_logrank_test

# for ous CT+PET+GTVt
h5_path = 'C:/Users/Windows User/Documents/UNI/M30-DV/ulrik/prediction_test_ous_ct_pet_tumor.h5'
csv_path = 'C:/Users/Windows User/Documents/UNI/M30-DV/HNC dataset/HNC tabular/OUS/OUS_D1.csv'
event_path = 'C:/Users/Windows User/Documents/UNI/M30-DV/HNC dataset/response_ous.csv'

with h5py.File(h5_path, 'r') as h5_file:
    patient_idx = pd.DataFrame(h5_file['patient_idx'][()], columns=['patient_id'])

    predicted_shape = h5_file['predicted'].shape[1]
    predicted_columns = [f'predicted_{i}' for i in range(predicted_shape)]
    predicted = pd.DataFrame(h5_file['predicted'][()], columns=predicted_columns)

    # For y dataset, take only the first 10 columns which correspond to predicted values
    y_columns = [f'y_{i}' for i in range(10)]
    y = pd.DataFrame(h5_file['y'][:, :10], columns=y_columns)

    h5_df = pd.concat([patient_idx, predicted, y], axis=1)


csv_df = pd.read_csv(csv_path)

event_df = pd.read_csv(event_path, delimiter=";")

# Select only the necessary columns from the additional .csv file
event_df_subset = event_df[['patient_id', 'event_OS', 'event_DFS']]

merged_df = pd.merge(csv_df, h5_df, on='patient_id')
# Merge the existing DataFrame with the additional .csv DataFrame
df = pd.merge(merged_df, event_df_subset, on='patient_id', how='left')

#calculated predicted survival time by cumulative adding the survved intervals
def calculate_survival_time(row):
    for i in range(10):
        if row[f'y_{i}'] == 0:
            return i
    return 10

df['survival_time_predicted'] = df.apply(calculate_survival_time, axis=1)


# split into groups based on covariate
covariate0 = df[df['uicc8_III-IV'] == 0]
covariate1 = df[df['uicc8_III-IV'] == 1]


plt.figure(figsize=(8, 8))

kmf1 = KaplanMeierFitter()
kmf1.fit(durations=covariate0['survival_time_predicted'], event_observed=covariate0['event_OS'], label='stage I-II')

kmf2 = KaplanMeierFitter()
kmf2.fit(durations=covariate1['survival_time_predicted'], event_observed=covariate1['event_OS'], label='stage III-IV')

ax = kmf1.plot_survival_function(ci_show=True, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})
kmf2.plot_survival_function(ax=ax, ci_show=True, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})

add_at_risk_counts(kmf1, kmf2, ax=ax)

plt.title('Overall Survival by Overall Stage of Disease for the CT+PET+GTVt model')
plt.tight_layout()
ax.legend(loc='lower left', fontsize=12)
ax.set_xlabel('Time Intervals', fontsize=12)
ax.set_ylabel('Survival probability', fontsize=12)
plt.savefig('C:/Users/Windows User/Documents/UNI/M30-DV/avhandling/m30-dv/images/KM_curev_CT_PET_tumor.png', dpi=300)

result_mlrt = pairwise_logrank_test(df['survival_time_predicted'],
                                    df['uicc8_III-IV'],
                                    df['event_OS'])

result_mlrt.print_summary()