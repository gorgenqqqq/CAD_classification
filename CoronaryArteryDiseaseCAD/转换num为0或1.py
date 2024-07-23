import pandas as pd

# 读取data_1.csv文件
input_file = 'data_1.csv'
data = pd.read_csv(input_file, header=None)

# 将最后一列不为0的值替换为1
data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: 1 if x != 0 else 0)


# 保存为新的CSV文件
output_file = 'data.csv'
data.to_csv(output_file, index=False, header=False)
