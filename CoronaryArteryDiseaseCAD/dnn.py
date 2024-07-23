import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
df = pd.read_csv('datafile.csv')

# 分离特征和目标变量
X = df.drop('num', axis=1).values  # 输入特征
y = df['num'].values  # 目标变量

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 目标变量转换为 PyTorch 张量
y = torch.tensor(y, dtype=torch.long)

# 拆分数据集为训练集和测试集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.numpy(), test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义深度神经网络模型
class DeepNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)  # 输出层的神经元数等于目标类别的数量
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 初始化模型、损失函数和优化器
input_dim = X_train_tensor.shape[1]
num_classes = len(torch.unique(y_train_tensor))
model = DeepNN(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(500):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{500}, Loss: {epoch_loss:.4f}')

# 评估模型
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# 计算 Accuracy
accuracy = accuracy_score(y_true, y_pred)

# 计算 Recall
recall = recall_score(y_true, y_pred, average='weighted')

# 计算 Precision
precision = precision_score(y_true, y_pred, average='weighted')

# 计算 Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# 计算 Specificity
specificity = tn / (tn + fp)

# 打印结果
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Specificity: {specificity:.4f}')
