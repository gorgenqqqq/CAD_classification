import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 选择指定的列
columns_to_keep = ['cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
df_selected = df[columns_to_keep]

# 保存到新的 CSV 文件
df_selected.to_csv('datafile.csv', index=False)
