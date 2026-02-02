---

# End-to-End Deep Learning Framework for Drug Discovery

## Introduction

본 프로젝트는 **AI 기반 신약 개발(AI-Driven Drug Discovery)**의 전 주기를 포괄하는 딥러닝 프레임워크입니다. 분자 물성 예측(Property Prediction), 3D 구조 분석(Conformation Analysis), 그리고 신규 후보 물질 생성(De Novo Design)을 수행하기 위해 **Geometric Deep Learning**과 **Generative Models**의 최신 방법론을 구현하였습니다.

단순한 모델 구현을 넘어, 실무 적용을 고려한 **Hyperparameter Tuning, Early Stopping, Modular Architecture**를 적용하여 코드의 재사용성과 모델의 일반화 성능(Robustness)을 확보하였습니다.

## Model Architectures & Methodologies

본 프레임워크는 연구 목적에 따라 총 4개의 핵심 모듈로 구성됩니다.

### 1. Molecular Property Prediction (Graph Neural Networks)

약물의 지질 친화도(Lipophilicity, LogP)를 정밀하게 예측하기 위한 회귀(Regression) 모델입니다.

* **GCN (Graph Convolutional Network):** 기본적인 분자 그래프 학습을 위한 Baseline 모델.
* **GAT (Graph Attention Network):** Self-Attention 메커니즘을 도입하여, 분자 내 주요 작용기(Functional Group) 간의 상호작용 가중치를 학습, 예측 정확도를 극대화하였습니다.

### 2. 3D Structure Analysis (Geometric Deep Learning)

분자의 2D 위상 정보뿐만 아니라, 3차원 공간 좌표(Atomic Coordinates)를 학습에 활용합니다.

* **SchNet:** Continuous-filter Convolution을 적용하여 원자 간의 거리 정보와 상호작용 에너지를 모델링합니다.

### 3. Generative Chemistry (De Novo Design)

신규 약물 후보 물질 생성을 위한 생성형 AI 모델을 구현하였습니다.

* **VGAE (Variational Graph Autoencoder):** 분자 그래프의 잠재 공간(Latent Space)을 학습하여 새로운 분자 구조를 생성합니다.
* **Diffusion Model (Score-based Generative Modeling):** 노이즈 제거 과정(Denoising Process)을 통해 안정적인 3D 분자 구조(Conformation)를 생성합니다.

### 4. Protein-Ligand Interaction

* **Binding Simulation:** 타겟 단백질(PDB)과 리간드 간의 결합 친화도(Binding Affinity)를 Radius Graph 기반으로 분석합니다.

## Performance Benchmarks

`MoleculeNet (Lipo)` 데이터셋을 기준으로 수행한 성능 평가 결과입니다. GAT 모델이 Baseline(GCN) 대비 유의미한 성능 향상을 보였습니다.

| Model | Task | Metric (Test) | Note |
| --- | --- | --- | --- |
| **GCN** | LogP Regression | MAE: 0.8275 | Baseline |
| **GAT** | LogP Regression | **MAE: 0.5924** | **SOTA-level Performance** |
| **SchNet** | 3D Energy Regression | MAE: 0.7743 | Geometric Learning |
| **VGAE** | Link Prediction | **AUC: 1.0000** | Graph Reconstruction |

## Repository Structure

```text
Drug-Discovery-AI/
├── 01_Property_Prediction/   # GNN (GCN, GAT) Training & Inference
├── 02_3D_Structure/          # SchNet Training
├── 03_Generative_AI/         # Generative Models (VGAE, Diffusion)
├── 04_Molecular_Interaction/ # Protein-Ligand Binding Simulation
├── docs/                     # Experiment Reports & Documentation
└── data/                     # Dataset Storage

```

## Usage & Reproduction

본 프로젝트는 `Python 3.8+` 및 `PyTorch Geometric` 환경에서 구동됩니다.

### Installation

```bash
pip install -r requirements.txt

```

### 1. Property Prediction

Graph Attention Network(GAT)를 이용한 물성 예측 모델 학습 및 추론:

```bash
# Training
python 01_Property_Prediction/train_gat.py

# Inference (Real-world Data)
python 01_Property_Prediction/predict.py

```

### 2. 3D Molecular Analysis

SchNet을 이용한 3D 구조 및 에너지 학습:

```bash
python 02_3D_Structure/train_schnet.py

```

### 3. Generative Models

Diffusion Model을 이용한 3D 분자 구조 생성:

```bash
# Training
python 03_Generative_AI/train_diffusion.py

# Generation (Inference)
python 03_Generative_AI/inference_diffusion.py

```

### 4. Interaction Simulation

특정 타겟(예: Gleevec)에 대한 결합 시뮬레이션:

```bash
python 04_Molecular_Interaction/binding_gleevec.py

```

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 5. Protein Design Demo (ProteinMPNN → ColabFold → Scoring)

단백질 디자인 직무에 맞춰, ProteinMPNN으로 서열을 디자인하고 ColabFold(AlphaFold2)로 구조를 예측한 뒤
pLDDT/RMSD 등의 지표로 후보를 랭킹하는 **Design→Predict→Validate** 데모를 추가했습니다.

- 자세한 사용 방법: `05_protein_design_demo/README.md`


