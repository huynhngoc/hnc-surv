import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

base_path = 'P:/REALTEK-HeadNeck-Project/Masteroppgaver_2024/Torjus/XAI/'

dfs_ous_smoothen_v2 = pd.read_csv(base_path + 'vargrad_dfs_ous.csv')
dfs_maastro_smoothen_v2 = pd.read_csv(base_path + 'vargrad_dfs_maastro.csv')

os_ous_smoothen_v2 = pd.read_csv(base_path + 'vargrad_os_ous.csv')
os_maastro_smoothen_v2 = pd.read_csv(base_path + 'vargrad_os_maastro.csv')



# region Check SUV histogram
suv_regions = ['zeros', '0_2', '2_4', '4_6', '6_8', '8_10', '10_over']
suv_df = dfs_ous_smoothen_v2[dfs_ous_smoothen_v2['quantile'] == 0.95][[
    f'all_suv_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

sum_suv_df = suv_df.sum()
hist_vals = []
for suv in suv_regions:
    hist_vals.append(
        sum_suv_df[f'all_suv_{suv}_sum'] / sum_suv_df[f'all_suv_{suv}_area'])

maastro_suv_df = dfs_maastro_smoothen_v2[dfs_maastro_smoothen_v2['quantile'] == 0.95][[
    f'all_suv_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

maastro_sum_suv_df = maastro_suv_df.sum()
maastro_hist_vals = []
for suv in suv_regions:
    maastro_hist_vals.append(
        maastro_sum_suv_df[f'all_suv_{suv}_sum'] / maastro_sum_suv_df[f'all_suv_{suv}_area'])

ax = sns.lineplot(x=suv_regions, y=hist_vals,
                  marker='o', lw=3, label='OUS')
ax = sns.lineplot(x=suv_regions, y=maastro_hist_vals,
                  marker='o', lw=3, label='MAASTRO')
ax.set_xticklabels([0, 2, 4, 6, 8, 10, '>10'])
ax.set_xlabel('SUV')
ax.set_ylabel('Mean VarGrad')
ax.set_title('DFS (PET only)')
plt.show()

# endregion SUV histogram


# region Check vargrad on tumor + node
dfs_smoothen_df = pd.concat([dfs_ous_smoothen_v2, dfs_maastro_smoothen_v2])
area_df = dfs_smoothen_df[dfs_smoothen_df['quantile'] == 0.95][['center', 'tumor_size', 'node_size',
                                                                'pt_tumor_all_sum', 'pt_node_all_sum', 'pt_outside_all_sum']]

area_df['tumor_sum'] = area_df['pt_tumor_all_sum']
area_df['node_sum'] =  area_df['pt_node_all_sum']
area_df['outside_sum'] = area_df['pt_outside_all_sum']
area_df['outside_size'] = 191*173*265 - area_df['tumor_size'] - area_df['node_size']


area_df_cal = area_df[['center', 'tumor_sum', 'tumor_size', 'node_sum',
                       'node_size', 'outside_sum', 'outside_size']].groupby('center').sum().reset_index()

area_df_cal['Tumor'] = area_df_cal['tumor_sum'] / \
    area_df_cal['tumor_size']
area_df_cal['Node'] = area_df_cal['node_sum'] / area_df_cal['node_size']
area_df_cal['Others'] = area_df_cal['outside_sum'] / \
    area_df_cal['outside_size']

ax = sns.barplot(area_df_cal[['center', 'Tumor', 'Node', 'Others']].melt(
    'center'), x='variable', y='value', hue='center', order=['Tumor', 'Node', 'Others'], hue_order=['OUS', 'MAASTRO'])
ax.set_xlabel('Area')
ax.set_ylabel('Mean VarGrad')
ax.set_title('DFS (PET only)')
plt.show()

# endregion


# region Check SUV histogram OS CT PET GTVp
suv_regions = ['zeros', '0_2', '2_4', '4_6', '6_8', '8_10', '10_over']
suv_df = os_ous_smoothen_v2[os_ous_smoothen_v2['quantile'] == 0.95][[
    f'all_suv_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

sum_suv_df = suv_df.sum()
hist_vals = []
for suv in suv_regions:
    hist_vals.append(
        sum_suv_df[f'all_suv_{suv}_sum'] / sum_suv_df[f'all_suv_{suv}_area'])

maastro_suv_df = os_maastro_smoothen_v2[os_maastro_smoothen_v2['quantile'] == 0.95][[
    f'all_suv_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

maastro_sum_suv_df = maastro_suv_df.sum()
maastro_hist_vals = []
for suv in suv_regions:
    maastro_hist_vals.append(
        maastro_sum_suv_df[f'all_suv_{suv}_sum'] / maastro_sum_suv_df[f'all_suv_{suv}_area'])

ax = sns.lineplot(x=suv_regions, y=hist_vals,
                  marker='o', lw=3, label='OUS')
ax = sns.lineplot(x=suv_regions, y=maastro_hist_vals,
                  marker='o', lw=3, label='MAASTRO')
ax.set_xticklabels([0, 2, 4, 6, 8, 10, '>10'])
ax.set_xlabel('SUV')
ax.set_ylabel('Mean VarGrad')
ax.set_title('OS (PET/CT + GTVp)', loc='left')
plt.show()

# endregion SUV histogram

# region Check HU histogram
suv_regions = ['zero', '0_16', '16_32', '32_48', '48_64',
               '64_80', '80_96', '96_112', '112_128',
               '128_144', '144_160', '160_176', '176_over']
suv_df = os_ous_smoothen_v2[os_ous_smoothen_v2['quantile'] == 0.95][[
    f'all_hu_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

sum_suv_df = suv_df.sum()
hist_vals = []
for suv in suv_regions:
    hist_vals.append(
        sum_suv_df[f'all_hu_{suv}_sum'] / sum_suv_df[f'all_hu_{suv}_area'])

maastro_suv_df = os_maastro_smoothen_v2[os_maastro_smoothen_v2['quantile'] == 0.95][[
    f'all_hu_{suv}_{stat}' for suv in suv_regions for stat in ['area', 'sum', 'total']]]

maastro_sum_suv_df = maastro_suv_df.sum()
maastro_hist_vals = []
for suv in suv_regions:
    maastro_hist_vals.append(
        maastro_sum_suv_df[f'all_hu_{suv}_sum'] / maastro_sum_suv_df[f'all_hu_{suv}_area'])

ax = sns.lineplot(x=suv_regions, y=hist_vals,
                  marker='o', lw=3, label='OUS')
ax = sns.lineplot(x=suv_regions, y=maastro_hist_vals,
                  marker='o', lw=3, label='MAASTRO')
# ax.set_xticklabels([-30, -14, 2, 18, 34, 50, 66, 82, 98, 114, 130, 146, 162])
ax.set_xticklabels([-30, None, 2, None, 34, None, 66,
                    None, 98, None, 130, None, 162])
ax.set_xlabel('HU')
ax.set_ylabel('Mean VarGrad')
ax.set_title('OS (PET/CT + GTVp)', loc='left')
plt.show()

# endregion SUV histogram


# region Check vargrad on tumor + node
os_smoothen_df = pd.concat([os_ous_smoothen_v2, os_maastro_smoothen_v2])
area_df = os_smoothen_df[os_smoothen_df['quantile'] == 0.95][['center', 'tumor_size', 'node_size', 'ct_tumor_all_sum',
                                                              'pt_tumor_all_sum', 'ct_node_all_sum', 'pt_node_all_sum', 'ct_outside_all_sum', 'pt_outside_all_sum']]

area_df['tumor_sum'] = (area_df['ct_tumor_all_sum'] +
                        area_df['pt_tumor_all_sum']) / 2
area_df['node_sum'] = (area_df['ct_node_all_sum'] +
                       area_df['pt_node_all_sum']) / 2
area_df['outside_sum'] = (
    area_df['ct_outside_all_sum'] + area_df['pt_outside_all_sum']) / 2
area_df['outside_size'] = 191*173*265 - \
    area_df['tumor_size'] - area_df['node_size']


area_df_cal = area_df[['center', 'tumor_sum', 'tumor_size', 'node_sum',
                       'node_size', 'outside_sum', 'outside_size']].groupby('center').sum().reset_index()

area_df_cal['Tumor'] = area_df_cal['tumor_sum'] / \
    area_df_cal['tumor_size']
area_df_cal['Node'] = area_df_cal['node_sum'] / area_df_cal['node_size']
area_df_cal['Others'] = area_df_cal['outside_sum'] / \
    area_df_cal['outside_size']

ax = sns.barplot(area_df_cal[['center', 'Tumor', 'Node', 'Others']].melt(
    'center'), x='variable', y='value', hue='center', order=['Tumor', 'Node', 'Others'], hue_order=['OUS', 'MAASTRO'])
ax.set_xlabel('Area')
ax.set_ylabel('Mean VarGrad')
sns.move_legend(ax, 'upper right', title=None,)
ax.set_title('OS (PET/CT + GTVp)', loc='left')
plt.show()

# endregion Check vargrad on tumor + node


########################################
### MISC

# check SUV correlation in DFS model
sns.violinplot(data=dfs_smoothen_df[['pid','suv_corr_raw', 'center']].groupby(['pid', 'center']).mean().reset_index().melt(
    ['center', 'pid']), x='value', y='variable', hue='center')
plt.show()


sns.violinplot(data=dfs_smoothen_df[['pid','suv_corr', 'center']].groupby(['pid', 'center']).mean().reset_index().melt(
    ['center', 'pid']), x='value', y='variable', hue='center')
plt.show()


check_suv_corr = dfs_smoothen_df[['pid','suv_corr_raw', 'suv_corr', 'center']].groupby(['pid', 'center']).mean().reset_index()
check_suv_corr[check_suv_corr.suv_corr < 0.2]
check_suv_corr[check_suv_corr.suv_corr_raw < 0.3]

# check SUV correlation in OS model
sns.violinplot(data=os_smoothen_df[['pid','suv_corr_raw', 'center']].groupby(['pid', 'center']).mean().reset_index().melt(
    ['center', 'pid']), x='value', y='variable', hue='center')
plt.show()


sns.violinplot(data=os_smoothen_df[['pid','suv_corr', 'center']].groupby(['pid', 'center']).mean().reset_index().melt(
    ['center', 'pid']), x='value', y='variable', hue='center')
plt.show()


# check_suv_corr = os_smoothen_df[['pid','suv_corr_raw', 'suv_corr', 'center']].groupby(['pid', 'center']).mean().reset_index()
# check_suv_corr[check_suv_corr.suv_corr < 0.2]
# check_suv_corr[check_suv_corr.suv_corr_raw < 0.3]
