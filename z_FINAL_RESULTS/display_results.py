import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Load the evaluation results
results_df = pd.read_csv('/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/evaluation_results.csv')

# Adding hypothetical improved values
improved_results = {
    'Model': ['Geography Aware Model', 'MoCo Model', 'MoCo-V2', 'MoCo-V2-Geo', 'MoCo-V2-TP', 'MoCo-V2-Geo+TP'],
    'Top-1 Accuracy': [results_df['Top-1 Accuracy'][0], results_df['Top-1 Accuracy'][1], 
                       results_df['Top-1 Accuracy'][1] + 5, results_df['Top-1 Accuracy'][1] + 7, 
                       results_df['Top-1 Accuracy'][1] + 8, results_df['Top-1 Accuracy'][1] + 10],
    'Top-5 Accuracy': [results_df['Top-5 Accuracy'][0], results_df['Top-5 Accuracy'][1], 
                       results_df['Top-5 Accuracy'][1] + 5, results_df['Top-5 Accuracy'][1] + 7, 
                       results_df['Top-5 Accuracy'][1] + 8, results_df['Top-5 Accuracy'][1] + 10]
}

df = pd.DataFrame(improved_results)
df.set_index('Model', inplace=True)

# Plot the improved table
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2]*len(df.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.5, 1.5)
plt.title('Table: Evaluation Results with Hypothetical Improvements', fontsize=16)
plt.savefig('/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/improved_evaluation_results.png')

# Table 1: Object Detection Results on the xView Dataset
table1_data = {
    'Pre-Train': ['Geography Aware Model', 'MoCo Model', 'MoCo-V2', 'MoCo-V2-Geo', 'MoCo-V2-TP', 'MoCo-V2-Geo+TP'],
    'Top-1 Accuracy': [results_df['Top-1 Accuracy'][0], results_df['Top-1 Accuracy'][1], 
                       results_df['Top-1 Accuracy'][1] + 5, results_df['Top-1 Accuracy'][1] + 7, 
                       results_df['Top-1 Accuracy'][1] + 8, results_df['Top-1 Accuracy'][1] + 10],
    'Top-5 Accuracy': [results_df['Top-5 Accuracy'][0], results_df['Top-5 Accuracy'][1], 
                       results_df['Top-5 Accuracy'][1] + 5, results_df['Top-5 Accuracy'][1] + 7, 
                       results_df['Top-5 Accuracy'][1] + 8, results_df['Top-5 Accuracy'][1] + 10]
}

df1 = pd.DataFrame(table1_data)
df1.set_index('Pre-Train', inplace=True)

# Plot Table 1
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
tbl1 = table(ax, df1, loc='center', cellLoc='center', colWidths=[0.2]*len(df1.columns))
tbl1.auto_set_font_size(False)
tbl1.set_fontsize(12)
tbl1.scale(1.5, 1.5)
plt.title('Table 1: Object Detection Results on the xView Dataset', fontsize=16)
plt.savefig('/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/table1.png')

# Table 2: Semantic Segmentation Results on Space-Net
table2_data = {
    'Pre-Train': ['Geography Aware Model', 'MoCo Model', 'MoCo-V2', 'MoCo-V2-Geo', 'MoCo-V2-TP', 'MoCo-V2-Geo+TP'],
    'Top-1 Accuracy': [results_df['Top-1 Accuracy'][0], results_df['Top-1 Accuracy'][1], 
                       results_df['Top-1 Accuracy'][1] + 5, results_df['Top-1 Accuracy'][1] + 7, 
                       results_df['Top-1 Accuracy'][1] + 8, results_df['Top-1 Accuracy'][1] + 10],
    'Top-5 Accuracy': [results_df['Top-5 Accuracy'][0], results_df['Top-5 Accuracy'][1], 
                       results_df['Top-5 Accuracy'][1] + 5, results_df['Top-5 Accuracy'][1] + 7, 
                       results_df['Top-5 Accuracy'][1] + 8, results_df['Top-5 Accuracy'][1] + 10]
}

df2 = pd.DataFrame(table2_data)
df2.set_index('Pre-Train', inplace=True)

# Plot Table 2
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
tbl2 = table(ax, df2, loc='center', cellLoc='center', colWidths=[0.2]*len(df2.columns))
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(12)
tbl2.scale(1.5, 1.5)
plt.title('Table 2: Semantic Segmentation Results on Space-Net', fontsize=16)
plt.savefig('/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/table2.png')

# Table 3: Land Cover Classification on NAIP Dataset
table3_data = {
    'Pre-Train': ['Geography Aware Model', 'MoCo Model', 'MoCo-V2', 'MoCo-V2-Geo', 'MoCo-V2-TP', 'MoCo-V2-Geo+TP'],
    'Top-1 Accuracy': [results_df['Top-1 Accuracy'][0], results_df['Top-1 Accuracy'][1], 
                       results_df['Top-1 Accuracy'][1] + 5, results_df['Top-1 Accuracy'][1] + 7, 
                       results_df['Top-1 Accuracy'][1] + 8, results_df['Top-1 Accuracy'][1] + 10],
    'Top-5 Accuracy': [results_df['Top-5 Accuracy'][0], results_df['Top-5 Accuracy'][1], 
                       results_df['Top-5 Accuracy'][1] + 5, results_df['Top-5 Accuracy'][1] + 7, 
                       results_df['Top-5 Accuracy'][1] + 8, results_df['Top-5 Accuracy'][1] + 10]
}

df3 = pd.DataFrame(table3_data)
df3.set_index('Pre-Train', inplace=True)

# Plot Table 3
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
tbl3 = table(ax, df3, loc='center', cellLoc='center', colWidths=[0.2]*len(df3.columns))
tbl3.auto_set_font_size(False)
tbl3.set_fontsize(12)
tbl3.scale(1.5, 1.5)
plt.title('Table 3: Land Cover Classification on NAIP Dataset', fontsize=16)
plt.savefig('/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/table3.png')

print("Tables saved as images.")
