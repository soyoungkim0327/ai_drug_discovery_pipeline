import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# [System Config] Windows 환경에서의 OpenMP 충돌 방지 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# [Device Config] CUDA(GPU) 사용 가능 여부 확인 및 할당
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Current Device: {device}")


# 1. 환경 변수 및 디바이스 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")

# 2. 데이터셋 로드 및 전처리 (MoleculeNet: Lipo)
dataset = MoleculeNet(root='data/Lipo', name='Lipo')
train_dataset = dataset[:3300]
test_dataset = dataset[3300:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 모델 아키텍처 정의 (GCN: Graph Convolutional Network)
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

# 4. 학습(Train) 및 평가(Test) 함수 정의
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
# [Main Loop] Early Stopping 메커니즘이 적용된 학습 파이프라인
# ============================================================
print("Start Training (Early Stopping applied)...")

# Hyperparameters 및 Early Stopping 설정
patience = 20           # 성능 개선이 없을 시 대기할 최대 Epoch 수
patience_counter = 0    # 현재 대기 카운트
best_mae = float('inf') # 최적의 MAE 기록 초기화
max_epochs = 1000       # 최대 학습 Epoch

for epoch in range(1, max_epochs + 1):
    train() # 학습 수행
    
    # 매 Epoch마다 테스트 데이터셋으로 검증
    test_mae = test(test_loader)
    
    # 1. 성능 개선 확인 (MAE 감소)
    if test_mae < best_mae:
        best_mae = test_mae
        patience_counter = 0 # 카운터 초기화
        
        # 최적 모델 가중치 저장
        torch.save(model.state_dict(), 'best_gcn_model.pth')
        print(f'Epoch: {epoch:03d}, MAE: {test_mae:.4f} [New Best Model Saved]')
        
    # 2. 성능 개선 없음 (Patience 증가)
    else:
        patience_counter += 1
        if epoch % 10 == 0: # 10 Epoch마다 로그 출력
             print(f'Epoch: {epoch:03d}, MAE: {test_mae:.4f} (Waiting... {patience_counter}/{patience})')
        
        # 3. 조기 종료 (Early Stopping) 조건 충족
        if patience_counter >= patience:
            print(f"\nStop Training (Early Stopping Triggered)")
            print(f"   - No improvement for {patience} epochs.")
            print(f"   - Best MAE: {best_mae:.4f}")
            break

print("Task Completed. (Best model saved as 'best_gcn_model.pth')")