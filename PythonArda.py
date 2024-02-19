# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path instructions for each dataFrame
file_path_cpu = 'C:\\Users\\Jordi Sabadell\\OneDrive - Office 365 Fontys\\3th Semester\\Arda2023\\Processor.csv'
file_path_os = 'C:\\Users\\Jordi Sabadell\\OneDrive - Office 365 Fontys\\3th Semester\\Arda2023\\OS.csv'
file_path_gpu = 'C:\\Users\\Jordi Sabadell\\OneDrive - Office 365 Fontys\\3th Semester\\Arda2023\\GPU.csv'
file_path_survey = 'C:\\Users\\Jordi Sabadell\\OneDrive - Office 365 Fontys\\3th Semester\\Arda2023\\ArdaSurvey.csv'

# Variables for reading each csv dataFrame 
df_os = pd.read_csv(file_path_os)
df_cpu = pd.read_csv(file_path_cpu, sep=';')
df_gpu = pd.read_csv(file_path_gpu, sep=';')
df_survey = pd.read_csv(file_path_survey)

# OS

# Melting and fixing the os steam list
steam_os_melted = df_os.melt(var_name='OS', value_name='Steam Usage Percentage')
steam_os_melted['OS'] = steam_os_melted['OS'].astype(str)
os_list = steam_os_melted.at[0, 'OS'].split(';')
steam_usage_percentages = steam_os_melted.loc[0, 'Steam Usage Percentage'].split(';')
split_steam_os_df = pd.DataFrame({
    'OS': os_list,
    'Steam Usage Percentage': steam_usage_percentages
})

# Calculating usage percentage of components in personal survey
os_usage_percentage = df_survey['User OS'].value_counts(normalize=True) * 100
# Renaming the column names to the DataFrame
personal_os_percentage = os_usage_percentage.reset_index()
personal_os_percentage.columns = ['OS', 'Personal Usage Percentage']
# Making sure OS calumns are type String
personal_os_percentage['OS'] = personal_os_percentage['OS'].astype(str)
# Map 'MacOS' and 'Linux' in personal_os_percentage to 'Other' column
personal_os_percentage['OS'] = personal_os_percentage['OS'].replace({'MacOS': 'Other', 'Linux': 'Other'})
if 'Other' not in split_steam_os_df['OS'].unique():
    split_steam_os_df = split_steam_os_df.append({'OS': 'Other', 'Steam Usage Percentage': 0}, ignore_index=True)
    
# Merge the two datasets
comparison_os_df = pd.merge(split_steam_os_df, personal_os_percentage, on='OS', how='outer')
comparison_os_df['OS'] = comparison_os_df['OS'].astype(str)
# Drop rows with NaN in 'OS' column
comparison_os_df = comparison_os_df.dropna(subset=['Steam Usage Percentage', 'Personal Usage Percentage'])

# Convert 'Steam Usage Percentage' and 'Personal Usage Percentage' to numeric, coercing errors to NaN
comparison_os_df['Steam Usage Percentage'] = pd.to_numeric(comparison_os_df['Steam Usage Percentage'], errors='coerce')
comparison_os_df['Personal Usage Percentage'] = pd.to_numeric(comparison_os_df['Personal Usage Percentage'], errors='coerce')
# Drop any rows that now have NaN in 'Steam Usage Percentage' or 'Personal Usage Percentage' after coercion
OS_comparison = comparison_os_df.dropna(subset=['Steam Usage Percentage', 'Personal Usage Percentage'])

# Plotting the comparison
plt.figure(figsize=(10, 6))
bar_width = 0.4
r1 = plt.bar(OS_comparison['OS'], OS_comparison['Steam Usage Percentage'], width=-bar_width, label='Steam Survey', alpha=0.6, align='edge')
r2 = plt.bar(OS_comparison['OS'], OS_comparison['Personal Usage Percentage'], width=bar_width, label='Personal Survey', alpha=0.6, align='edge')
for rect in r1+r2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')
windows_7_index = OS_comparison[OS_comparison['OS'] == 'Windows 7'].index[0]
windows_8_index = OS_comparison[OS_comparison['OS'] == 'Windows 8'].index[0]
plt.text(windows_7_index - bar_width/2, OS_comparison.at[windows_7_index, 'Steam Usage Percentage'], f"{OS_comparison.at[windows_7_index, 'Steam Usage Percentage']:.2f}", ha='center', va='bottom')
plt.text(windows_8_index + bar_width/2, OS_comparison.at[windows_8_index, 'Personal Usage Percentage'], f"{OS_comparison.at[windows_8_index, 'Personal Usage Percentage']:.2f}", ha='center', va='bottom')
plt.xlabel('Operating System', fontsize=12)
plt.ylabel('Usage Percentage', fontsize=12)
plt.title('Comparison of OS Usage: Steam Survey vs Personal Survey', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.tight_layout()
plt.show()

# GPU

# Calculating the usage percentage of GPU in personal survey
gpu_percent = df_survey['User GPU'].value_counts(normalize=True) * 100
personal_gpu_percent = gpu_percent.reset_index()
personal_gpu_percent.columns = ['User GPU', 'GPU Usage Percentage']

#First, find the GPU models from the personal survey that do not match any in the steam survey
unique_gpus_personal = set(personal_gpu_percent['User GPU']) - set(df_gpu['User GPU'])

# Map these unique personal survey GPU models to 'Other'
personal_gpu_percent['User GPU'] = personal_gpu_percent['User GPU'].replace(list(unique_gpus_personal), 'Other')

# Ensure there's an 'Other' category in the steam survey dataframe
if 'Other' not in df_gpu['User GPU'].values:
    df_gpu = df_gpu.append({'User GPU': 'Other', 'Usage': 0}, ignore_index=True)

# Replace "NVIDIA GeForce" with an empty string in the 'User GPU' column of both DataFrames
df_gpu['User GPU'] = df_gpu['User GPU'].str.replace("NVIDIA GeForce ", "")
personal_gpu_percent['User GPU'] = personal_gpu_percent['User GPU'].str.replace("NVIDIA GeForce ", "")

# Merge the two DataFrames on the 'User GPU' column
comparison_gpu_df = pd.merge(df_gpu, personal_gpu_percent, on='User GPU', how='outer')
other_variations = ['M2', 'M1 Pro']  
comparison_gpu_df['User GPU'] = comparison_gpu_df['User GPU'].replace(other_variations, 'Other')
comparison_gpu_df = comparison_gpu_df.groupby('User GPU', as_index=False).sum()

# "Other" at the end
other_row = comparison_gpu_df[comparison_gpu_df['User GPU'] == 'Other']
non_other_df = comparison_gpu_df[comparison_gpu_df['User GPU'] != 'Other']
final_gpu_df = pd.concat([non_other_df, other_row]).reset_index(drop=True)
print(final_gpu_df)
# Plotting the comparison
bar_width = 0.4
r1 = range(len(final_gpu_df['User GPU']))
r2 = [x + bar_width for x in r1]
plt.figure(figsize=(15, 8))
plt.bar(r1, final_gpu_df['Usage'], width=bar_width, label='Steam Survey', alpha=0.6)
plt.bar(r2, final_gpu_df['GPU Usage Percentage'], width=bar_width, label='Personal Survey', alpha=0.6)
plt.xlabel('GPU Model')
plt.ylabel('Usage Percentage')
plt.title('Comparison of GPU Usage: Steam Survey vs Personal Survey')
plt.xticks([r + bar_width/2 for r in r1], final_gpu_df['User GPU'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# CPU brand between Steam and personal survey

def categorize_cpu(brand):
    if 'Intel' in brand:
        return 'Intel'
    elif 'Ryzen' in brand:
        return 'AMD'
    elif 'M1 Pro' in brand or 'M2' in brand:
        return 'Other'
    else:
        return 'Other'
    
df_survey['CPU Brand'] = df_survey['User CPU '].apply(categorize_cpu)

# Calculating the usage percentage of CPU in personal survey
cpu_brand_usage = df_survey['CPU Brand'].value_counts(normalize=True) * 100
cpu_brand_usage_df = cpu_brand_usage.reset_index()
cpu_brand_usage_df.columns = ['CPU Brand', 'Usage Percentage']

# Merge the two DataFrames on the 'CPU Brand' column
comparison_cpu_df = pd.merge(df_cpu, cpu_brand_usage_df, on='CPU Brand', how='outer')

# Plotting the comparison
bar_width = 0.4
r1 = range(len(comparison_cpu_df['CPU Brand']))
r2 = [x + bar_width for x in r1]
plt.figure(figsize=(10, 6))
bars1 = plt.bar(r1, comparison_cpu_df['Usage'], width=bar_width, label='Steam Survey', alpha=0.6)
bars2 = plt.bar(r2, comparison_cpu_df['Usage Percentage'], width=bar_width, label='Personal Survey', alpha=0.6)
plt.xlabel('CPU Brand')
plt.ylabel('Usage Percentage')
plt.title('Comparison of CPU Usage: Steam Survey vs Personal Survey')
plt.xticks([r + bar_width/2 for r in r1], comparison_cpu_df['CPU Brand'], rotation=45)
plt.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 2), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# CPU models 

# Calculating CPU usage percentage from personal survey
cpu_usage_counts = df_survey['User CPU '].value_counts(normalize=True) * 100
cpu_usage_df = cpu_usage_counts.reset_index()
cpu_usage_df.columns = ['User CPU', 'Usage Percentage']
cpu_usage_df.sort_values('Usage Percentage', ascending=False, inplace=True)

# Plotting 
plt.figure(figsize=(10, 6))
bars = plt.bar(cpu_usage_df['User CPU'], cpu_usage_df['Usage Percentage'], color='skyblue')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
plt.xlabel('CPU Model')
plt.ylabel('Usage Percentage (%)')
plt.title('Usage Percentage of CPU Models in Personal Survey')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# OS level of satisfaction

os_satisfaction = df_survey.groupby('User OS')['Level of satisfaction'].mean().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(os_satisfaction['User OS'], os_satisfaction['Level of satisfaction'], color='skyblue')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
plt.xlabel('Operating System')
plt.ylabel('Average Satisfaction Level')
plt.title('Average Satisfaction by Operating System')
plt.xticks(rotation=45)
plt.show()

# GPU level of satisfaction

response_counts = df_survey['User GPU'].value_counts()
df_survey = df_survey.merge(response_counts.rename('Response Count'), left_on='User GPU', right_index=True)
df_survey['Weighted Satisfaction'] = df_survey['Level of satisfaction'] * df_survey['Response Count']
weighted_satisfaction_gpu = df_survey.groupby('User GPU').apply(
    lambda x: np.average(x['Level of satisfaction'], weights=x['Response Count'])
).reset_index(name='Weighted Satisfaction')  
print(weighted_satisfaction_gpu)

#gpu_satisfaction = df_survey.groupby('User GPU')['Level of satisfaction'].mean().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(10, 6))
plt.bar(weighted_satisfaction_gpu['User GPU'], weighted_satisfaction_gpu['Weighted Satisfaction'], color='skyblue')
plt.xlabel('GPU Model')
plt.ylabel('Weighted Average Satisfaction Level')
plt.title('Weighted Average Satisfaction by GPU Model')
plt.xticks(rotation=90)
plt.show()

# CPU level of satisfaction

cpu_satisfaction = df_survey.groupby('User CPU ')['Level of satisfaction'].mean().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(cpu_satisfaction['User CPU '], cpu_satisfaction['Level of satisfaction'], color='salmon')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
plt.xlabel('CPU Model')
plt.ylabel('Average Satisfaction Level')
plt.title('Average Satisfaction by CPU Model')
plt.xticks(rotation=90)  
plt.show()



