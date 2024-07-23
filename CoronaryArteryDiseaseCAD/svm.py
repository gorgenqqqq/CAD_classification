import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# 读取数据
df = pd.read_csv('datafile.csv')

# 分离特征和目标变量
X = df.drop('num', axis=1)  # 输入特征
y = df['num']  # 目标变量

# 拆分数据集为训练集和测试集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化并训练 SVM 模型
model = SVC(kernel='linear')  # 可以选择其他核函数，如 'rbf' 等
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算 Accuracy
accuracy = accuracy_score(y_test, y_pred)

# 计算 Recall
recall = recall_score(y_test, y_pred)

# 计算 Precision
precision = precision_score(y_test, y_pred)

# 计算 Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 计算 Specificity
specificity = tn / (tn + fp)

# 打印结果
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Specificity: {specificity:.4f}')
