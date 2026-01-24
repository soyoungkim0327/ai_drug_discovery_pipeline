import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# [System Config] Windows 환경에서의 OpenMP 라이브러리 충돌 방지 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# [Device Config] CUDA(GPU) 가용성 확인 및 디바이스 할당
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Current Device: {device}")
# (CUDA가 출력되면 GPU 가속이 활성화된 상태입니다)


# 1. Environment Configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")

# 2. Data Preparation (Load MoleculeNet: Lipo dataset)
dataset = MoleculeNet(root='data/Lipo', name='Lipo')
train_dataset = dataset[:3300]
test_dataset = dataset[3300:]

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Model Architecture Definition (GCN)
class GCNModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)

model = GCNModel(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Define Training and Evaluation Functions
def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = F.mse_loss(out.squeeze(), data.y.squeeze()) 
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch)
        error += (out.squeeze() - data.y.squeeze()).abs().sum().item()
    return error / len(loader.dataset)

# ============================================================
# [Main Loop] Training Pipeline with Early Stopping Mechanism
# ============================================================
print("Start Training (Early Stopping applied)...")

# Hyperparameters 및 Early Stopping 설정 변수
patience = 20           # 성능 개선 부재 시 허용할 최대 Epoch 수 (Patience Threshold)
patience_counter = 0    # 현재 Patience 카운터
best_mae = float('inf') # 최적 MAE 기록 초기화
max_epochs = 1000       # 최대 학습 Epoch 설정

for epoch in range(1, max_epochs + 1):
    train() # Training Step 수행
    
    # 매 Epoch 종료 후 Test Set으로 성능 평가
    test_mae = test(test_loader)
    
    # 1. 성능 개선 확인 (MAE 감소)
    if test_mae < best_mae:
        best_mae = test_mae
        patience_counter = 0 # Patience 카운터 초기화
        
        # 최적 모델 가중치 저장 (Save Best Model Weights)
        torch.save(model.state_dict(), 'best_gcn_model.pth')
        print(f'Epoch: {epoch:03d}, MAE: {test_mae:.4f} [New Best Model Saved]')
        
    # 2. 성능 개선 없음 (Patience 증가)
    else:
        patience_counter += 1
        if epoch % 10 == 0: # 로그 가독성을 위해 10 Epoch마다 상태 출력
             print(f'Epoch: {epoch:03d}, MAE: {test_mae:.4f} (Waiting... {patience_counter}/{patience})')
        
        # 3. 조기 종료 조건 충족 (Early Stopping Triggered)
        if patience_counter >= patience:
            print(f"\nStop Training (Early Stopping Triggered)")
            print(f"   - No improvement for {patience} epochs.")
            print(f"   - Best MAE: {best_mae:.4f}")
            break

print("Task Completed. (Best model saved as 'best_gcn_model.pth')")