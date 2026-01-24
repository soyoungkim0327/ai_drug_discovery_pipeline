
---

### 파일명: `results_report.md`

```markdown
# Technical Report: Experimental Results & Model Benchmarks

본 문서는 **Drug AI Project**의 모델별 학습 수렴 과정, 성능 지표(Metrics), 그리고 최종 추론 결과를 기록한 기술 리포트입니다. 모든 실험은 과적합(Overfitting) 방지 및 일반화 성능 확보를 위해 `Early Stopping` 전략을 적용하여 수행되었습니다.

## 1. Performance Summary (성능 요약)

아래 표는 각 모델 아키텍처별 최종 성능과 수행 결과를 요약한 것입니다.

| Project | Model Architecture | Task Type | Best Metric | Status | Output File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Property Pred** | **GAT** (Graph Attention) | Regression (LogP) | **MAE: 0.5924** | Converged | `best_gat_model.pth` |
| | **GCN** (Baseline) | Regression (LogP) | MAE: 0.8275 | Converged | `best_gcn_model.pth` |
| **2. 3D Structure** | **SchNet** (Continuous Filter) | 3D Regression | MAE: 0.7743 | Converged | `best_schnet_model.pth` |
| **3. Generative** | **VGAE** (Var. Autoencoder) | Link Prediction | AUC: 1.0000 | Converged | `best_vgae_model.pth` |
| **4. Interaction** | **Binding Inference** | Interaction Analysis | N/A (Inference) | Completed | `pdb_comparison.png` |
| **5. Diffusion** | **Diffusion** (Score-based) | Point Cloud Gen | Loss: 0.0323 | Converged | `diffusion_process.png` |

---

## 2. Detailed Experiment Logs (상세 학습 로그)

다음은 각 모델의 학습 과정 중 주요 체크포인트와 수렴 시점을 기록한 상세 로그입니다. (가독성을 위해 시스템 경로는 상대 경로 `./`로 표기하였습니다.)

### 1. Molecular Property Prediction (LogP)

#### 1-1. Baseline Model: GCN (Graph Convolutional Network)
* **Configuration:** Patience=20, Max Epochs=1000
* **Result:** Baseline 모델로서 수렴하였으나, 복잡한 분자 구조의 특징 추출에 한계가 있음 (MAE 0.8275).

```bash
(drug_ai) PS ./Drug-Discovery-AI> python 1-1.GCN_LogP_Train.py
Current Device: cuda
Start Training (Early Stopping applied)...

Epoch: 001, MAE: 0.9448 [New Best Model Saved]
Epoch: 005, MAE: 0.9261 [New Best Model Saved]
Epoch: 010, MAE: 0.9526 (Patience: 5/20)
Epoch: 020, MAE: 0.9107 [New Best Model Saved]
Epoch: 030, MAE: 0.8805 [New Best Model Saved]
Epoch: 040, MAE: 0.8731 (Patience: 1/20)
Epoch: 055, MAE: 0.8275 [New Best Model Saved]
Epoch: 070, MAE: 0.8440 (Patience: 15/20)

Early Stopping Triggered.
   - No improvement for 20 epochs.
   - Best MAE: 0.8275
Task Completed. Best model saved to 'best_gcn_model.pth'.

```

#### 1-2. Advanced Model: GAT (Graph Attention Network)

* **Configuration:** Patience=20, Max Epochs=1000
* **Result:** Self-Attention 메커니즘을 도입하여 GCN 대비 **약 28% 성능 향상** (MAE 0.5924)을 달성함.

```bash
(drug_ai) PS ./Drug-Discovery-AI> python 1-2.GAT_LogP_Train.py
Current Device: cuda
Start GAT Training (Early Stopping applied)...

Epoch: 001, MAE: 1.0196 [New Best Model Saved]
Epoch: 010, MAE: 0.9046 [New Best Model Saved]
Epoch: 025, MAE: 0.8055 [New Best Model Saved]
Epoch: 053, MAE: 0.7216 [New Best Model Saved]
Epoch: 091, MAE: 0.6719 [New Best Model Saved]
Epoch: 128, MAE: 0.6369 [New Best Model Saved]
Epoch: 164, MAE: 0.6138 [New Best Model Saved]
Epoch: 177, MAE: 0.5924 [New Best Model Saved]
Epoch: 190, MAE: 0.6220 (Patience: 13/20)

Early Stopping Triggered.
   - No improvement for 20 epochs.
   - Best MAE: 0.5924
Task Completed. Best model saved to 'best_gat_model.pth'.

```

---

### 2. 3D Molecular Representation (SchNet)

* **Objective:** 2D SMILES 데이터를 3D Conformer로 변환하여 공간적 상호작용(Interaction)을 포함한 물성 예측.
* **Result:** 학습 초기 Loss가 빠르게 감소하며 안정적으로 수렴함.

```bash
(drug_ai) PS ./Drug-Discovery-AI> python 2.SchNet_3D_Train.py
Current Device: cuda
Start 3D SchNet Training (Early Stopping applied)...

Epoch: 001, Loss: 1.8883, Test MAE: 1.0745 [New Best Model Saved]
Epoch: 006, Loss: 1.2828, Test MAE: 0.9420 [New Best Model Saved]
Epoch: 017, Loss: 1.0438, Test MAE: 0.8472 [New Best Model Saved]
Epoch: 028, Loss: 0.5558, Test MAE: 0.8238 [New Best Model Saved]
Epoch: 039, Loss: 0.2282, Test MAE: 0.7743 [New Best Model Saved]
Epoch: 045, Loss: 0.1300, Test MAE: 0.8256 (Patience: 6/10)

Early Stopping Triggered. (Best MAE: 0.7743)
3D Training Completed. Model saved to 'best_schnet_model.pth'.

```

---

### 3. Generative AI: VGAE (Variational Graph Autoencoder)

* **Objective:** 잠재 공간(Latent Space) 학습을 통한 분자 그래프 구조의 생성 및 복원.
* **Result:** AUC 1.0000 도달 (데이터셋 내 분자 연결 구조를 완벽하게 복원함).

```bash
(drug_ai) PS ./Drug-Discovery-AI> python 3.VGAE_Gen_Train.py
Current Device: cuda
Start VGAE Training (Early Stopping applied)...

Epoch: 001, Loss: 46.6963, AUC: 1.0000 [New Best Model Saved]
Epoch: 010, Loss: 4.7879, AUC: 0.7500 (Patience: 9/20)
Epoch: 020, Loss: 2.9759, AUC: 1.0000 (Patience: 19/20)

Early Stopping Triggered. (Best AUC: 1.0000)
Generative Model Training Completed. Model saved to 'best_vgae_model.pth'.

```

---

### 4. Generative AI: Diffusion Model

* **Objective:** 노이즈(Noise)로부터 유의미한 데이터 포인트(Coordinate)를 복원하는 Denoising Process 구현.
* **Result:** Loss가 0.03 수준까지 수렴하여 성공적인 좌표 생성 능력을 검증함.

```bash
(drug_ai) PS ./Drug-Discovery-AI> python 5.Diffusion_Train.py
Start Diffusion Model Training (Early Stopping applied)...

Epoch 020, Loss: 0.213448 [New Best Model Saved]
...
Early Stopping Triggered.
   - Stopped at epoch 112. (Min Loss: 0.032310)

Training Completed. Starting Generation Process...
Model loaded: best_diffusion.pth
Generated Coordinates: [[0.9128584 0.8977687]]

Visualizing Diffusion Process...
Result saved: 'diffusion_early_stopping.png'

```

```

```