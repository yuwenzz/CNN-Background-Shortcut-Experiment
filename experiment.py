import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# 1. 固定随机种子，确保实验完全可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


# 2. 构建合成数据集（模拟“白底椅子 vs 深底非椅子”）
def create_image(obj_type, bg_type):
    img = np.zeros((3, 32, 32), dtype=np.float32)
    if bg_type == 'white':
        img[:, :, :] = np.random.uniform(0.92, 1.0, (3, 32, 32))
    else:
        img[:, :, :] = np.random.uniform(0.0, 0.15, (3, 32, 32))

    dx = random.randint(-3, 3)
    dy = random.randint(-3, 3)
    obj_color = np.array([0.45, 0.30, 0.20])[:, None, None]

    if obj_type == 'chair':
        img[:, 8 + dy:22 + dy, 10 + dx:13 + dx] = obj_color
        img[:, 19 + dy:22 + dy, 10 + dx:20 + dx] = obj_color
        img[:, 22 + dy:27 + dy, 10 + dx:12 + dx] = obj_color
        img[:, 22 + dy:27 + dy, 18 + dx:20 + dx] = obj_color
    else:
        img[:, 13 + dy:16 + dy, 7 + dx:25 + dx] = obj_color
        img[:, 16 + dy:26 + dy, 14 + dx:18 + dx] = obj_color

    return img


def build_datasets(n_train=400, n_test=100):
    train_x, train_y = [], []
    for _ in range(n_train // 2):
        train_x.append(create_image('chair', 'white'))
        train_y.append(1)
        train_x.append(create_image('table', 'dark'))
        train_y.append(0)

    test_x, test_y = [], []
    for _ in range(n_test // 2):
        test_x.append(create_image('chair', 'dark'))
        test_y.append(1)
        test_x.append(create_image('table', 'white'))
        test_y.append(0)

    return (
        TensorDataset(torch.tensor(np.array(train_x)), torch.tensor(train_y)),
        TensorDataset(torch.tensor(np.array(test_x)), torch.tensor(test_y))
    )


# 3. 定义标准三层 CNN 模型
class Simple3LayerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# 4. 执行训练与分布偏移评估
train_ds, test_ds = build_datasets(n_train=400, n_test=100)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = Simple3LayerCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

train_accs, test_accs = [], []
epochs = 10

print("=== 开始训练 3-Layer CNN ===")
for epoch in range(1, epochs + 1):
    model.train()
    correct, total = 0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    train_acc = correct / total * 100
    train_accs.append(train_acc)

    model.eval()
    t_correct, t_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            t_correct += (out.argmax(1) == y).sum().item()
            t_total += y.size(0)
    test_acc = t_correct / t_total * 100
    test_accs.append(test_acc)
    print(f"Epoch {epoch:02d}/{epochs:02d} | Biased Train Acc: {train_acc:6.2f}% | Shift Test Acc: {test_acc:6.2f}%")

# 5. 输出单样本预测结果
print("\n=== 单样本诊断预测 (Actual CNN Predictions) ===")
model.eval()
labels_map = {0: "Non-Chair", 1: "Chair"}

with torch.no_grad():
    sample_chair_dark = torch.tensor(create_image('chair', 'dark')).unsqueeze(0)
    prob1 = F.softmax(model(sample_chair_dark), dim=1)[0]
    pred1 = prob1.argmax().item()
    print(f"Sample [Chair in Dark/Room] -> Pred: {labels_map[pred1]} (Confidence: {prob1[pred1] * 100:.1f}%)")

    sample_table_white = torch.tensor(create_image('table', 'white')).unsqueeze(0)
    prob2 = F.softmax(model(sample_table_white), dim=1)[0]
    pred2 = prob2.argmax().item()
    print(f"Sample [Table on White]     -> Pred: {labels_map[pred2]} (Confidence: {prob2[pred2] * 100:.1f}%)")

# 6. 保存模型与曲线图
torch.save(model.state_dict(), "cnn_shortcut_model.pth")
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), train_accs, marker='o', label='Biased Train Acc (%)')
plt.plot(range(1, epochs + 1), test_accs, marker='s', linestyle='--', label='Distribution Shift Test Acc (%)')
plt.title('3-Layer CNN: Training Fit vs. Distribution Shift')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.ylim(-5, 105)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=200)
print("\n已在项目目录下生成权重文件 'cnn_shortcut_model.pth' 和曲线图 'training_curve.png'")