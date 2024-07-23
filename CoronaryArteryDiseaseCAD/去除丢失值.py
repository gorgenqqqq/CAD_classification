import pandas as pd

# 读取processed.cleveland.data文件
input_file = 'processed.cleveland.data'
data = pd.read_csv(input_file, header=None)

# 计算包含'?'的行数
num_missing = data.isin(['?']).sum().sum()
print(f"数据中包含 {num_missing} 个 '?'")

# 去除包含'?'的行
cleaned_data = data[~data.isin(['?']).any(axis=1)]

# 保存为新的CSV文件
output_file = 'datafile.csv'
cleaned_data.to_csv(output_file, index=False, header=False)

print(f"处理后的数据已保存到 {output_file}")
