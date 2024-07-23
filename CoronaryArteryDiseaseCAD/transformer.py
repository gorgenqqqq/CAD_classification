import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        encoded_feature = self.tokenizer.encode_plus(
            str(feature),  # Convert feature to string if necessary
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded_feature['input_ids'].flatten(),
            'attention_mask': encoded_feature['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 定义模型
class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TransformerClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token output
        logits = self.fc(cls_output)
        return logits


# 读取数据
df = pd.read_csv('datafile.csv')

# 分离特征和目标变量
X = df.drop('num', axis=1)  # 特征
y = df['num']  # 目标变量

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 现在 X 是数值型数据

# 拆分数据集为训练集和测试集（80% 训练集，20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 重设索引
X_train = pd.DataFrame(X_train).reset_index(drop=True)
X_test = pd.DataFrame(X_test).reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# 设置 tokenizer 和数据集
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
max_length = 128
train_dataset = CustomDataset(X_train.values, y_train.values, tokenizer, max_length)
test_dataset = CustomDataset(X_test.values, y_test.values, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型
num_classes = len(set(y))
model = TransformerClassifier(model_name, num_classes)

# 设置优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 训练模型
model.train()
for epoch in range(3):  # 训练轮数
    for batch in train_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 测试模型
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算 Accuracy
accuracy = accuracy_score(all_labels, all_preds)

# 计算 Recall
recall = recall_score(all_labels, all_preds, average='weighted')

# 计算 Precision
precision = precision_score(all_labels, all_preds, average='weighted')

# 计算 Confusion Matrix
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

# 计算 Specificity
specificity = tn / (tn + fp)

# 打印结果
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Specificity: {specificity:.4f}')
