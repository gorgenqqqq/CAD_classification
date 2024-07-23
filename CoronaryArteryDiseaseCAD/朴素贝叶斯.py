import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# 读取数据
df = pd.read_csv('datafile.csv')

# 分离特征和目标变量
X = df.drop('num', axis=1).values  # 输入特征
y = df['num'].values  # 目标变量

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集为训练集和测试集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化朴素贝叶斯分类器
model = GaussianNB()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算 Accuracy
accuracy = accuracy_score(y_test, y_pred)

# 计算 Recall
recall = recall_score(y_test, y_pred, average='weighted')

# 计算 Precision
precision = precision_score(y_test, y_pred, average='weighted')

# 计算 Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 计算 Specificity
specificity = tn / (tn + fp)

# 打印结果
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Specificity: {specificity:.4f}')
