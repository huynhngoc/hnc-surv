import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme()

data_ous = {
    "Model": ["CT+PET+GTVp", "PET+GTVp", "PET+GTVp+GTVn", "PET", "CT+PET", "CT+PET+GTVp+GTVn", "CT",
              "PET", "PET+GTVp", "CT+PET+GTVp", "CT+PET"],
    "C-index": [0.74, 0.72, 0.71, 0.70, 0.66, 0.66, 0.61,
               0.62, 0.60, 0.59, 0.59],
    "AUC": [0.69, 0.68, 0.67, 0.63, 0.61, 0.62, 0.56,
            0.55, 0.54, 0.54, 0.54],
    "IBS": [0.16, 0.16, 0.16, 0.17, 0.18, 0.17, 0.18,
            0.23, 0.24, 0.23, 0.23],
    "Survival Type": ["OS", "OS", "OS", "OS", "OS", "OS", "OS",
                      "DFS", "DFS", "DFS", "DFS"]
}

data_maastro = {
    "Model": ["CT+PET+GTVp", "PET+GTVp", "PET+GTVp+GTVn", "PET", "CT+PET", "CT+PET+GTVp+GTVn", "CT",
              "PET", "PET+GTVp", "CT+PET+GTVp", "CT+PET"],
    "C-index": [0.68, 0.65, 0.66, 0.67, 0.63, 0.69, 0.62,
               0.67, 0.61, 0.63, 0.65],
    "AUC": [0.68, 0.65, 0.64, 0.64, 0.62, 0.67, 0.60,
            0.63, 0.63, 0.64, 0.65],
    "IBS": [0.17, 0.17, 0.17, 0.17, 0.17, 0.16, 0.18,
            0.21, 0.22, 0.21, 0.21],
    "Difference in C-index": [-0.06, -0.07, -0.05, -0.03, -0.03, 0.03, 0.01,
                             0.05, 0.01, 0.04, 0.06],
    "Survival Type": ["OS", "OS", "OS", "OS", "OS", "OS", "OS",
                      "DFS", "DFS", "DFS", "DFS"]
}

df = pd.DataFrame(data_ous)
df_maastro = pd.DataFrame(data_maastro)

# Splitting the dataset into OS and DFS for OUS and MAASTRO
df_os = df[df["Survival Type"] == "OS"]
df_dfs = df[df["Survival Type"] == "DFS"]
df_os_maastro = df_maastro[df_maastro["Survival Type"] == "OS"]
df_dfs_maastro = df_maastro[df_maastro["Survival Type"] == "DFS"]

models = df['Model'].unique()
palette = sns.color_palette(n_colors=len(models))
model_palette = {model: color for model, color in zip(sorted(models), palette)}

def create_and_save_plots(df, df_name, survival_type, metrics):
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x='Model', y=metric, palette=model_palette)
        plt.title(f'{df_name} {metric} on {survival_type}', fontsize=20)
        plt.xticks(rotation=45)
        plt.xlabel('')
        plt.ylabel(metric, fontsize=16)
        # Adding the text labels above the bars
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')
        plt.tight_layout()
        plt.savefig(f'C:/Users/Windows User/Documents/UNI/M30-DV/avhandling/m30-dv/images/preds/{df_name}_{survival_type}_{metric.replace(" ", "_").lower()}.png')
        plt.close()

# OUS dataset
create_and_save_plots(df_os, "OUS", "OS", ["C-index", "AUC", "IBS"])
create_and_save_plots(df_dfs, "OUS", "DFS", ["C-index", "AUC", "IBS"])

# MAASTRO dataset
create_and_save_plots(df_os_maastro, "MAASTRO", "OS", ["C-index", "AUC", "IBS", "Difference in C-index"])
create_and_save_plots(df_dfs_maastro, "MAASTRO", "DFS", ["C-index", "AUC", "IBS", "Difference in C-index"])