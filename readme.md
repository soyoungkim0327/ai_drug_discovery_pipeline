---

# End-to-End Deep Learning Framework for Drug Discovery

## Introduction

본 프로젝트는 **AI 기반 신약 개발(AI-Driven Drug Discovery)**의 전 주기를 "데모 파이프라인" 형태로 연결한 딥러닝 프레임워크입니다.

- 분자 물성 예측(Property Prediction)
- 3D 구조 기반 예측(Conformation / Geometric Deep Learning)
- 생성형 모델 기반 구조 생성(Generative Models)
- 단백질-리간드 상호작용(Protein-Ligand Interaction) 분석

실험 재현성과 코드 품질을 위해 **train/val/test 분리**, **early stopping (VAL 기준)**, **robust path handling**, **metadata 로그(JSON)**를 포함하도록 정리했습니다.

## Model Architectures & Methodologies

### 1. Molecular Property Prediction (GNN)
- **GCN**: Baseline 회귀 모델
- **GAT**: Attention 기반 회귀 모델

### 2. 3D Structure Analysis (Geometric Deep Learning)
- **SchNet**: 원자 좌표 기반 연속 필터 컨볼루션

### 3. Generative Models
- **VGAE**: 그래프 링크 예측/복원 기반 잠재공간 학습 (※ 단일 그래프 데모)
- **Diffusion (Toy)**: 2D 포인트에서의 denoising trajectory 데모

### 4. Protein-Ligand Interaction
- **PDB 기반 거리/반경 그래프 기반 분석** (데모)

## Repository Structure

```text
ai_drug_discovery_pipeline/
├── 01_property_prediction/         # GNN Training & Inference
├── 02_3d_structure/                # SchNet Training & Inference
├── 03_generative_ai/               # VGAE, Diffusion
├── 04_molecular_interaction/       # PDB Interaction Demo
├── docs/                           # Reports
├── data/                           # Cached datasets
├── install_guide.md
├── readme.md
└── result.md
```

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### 1) Property Prediction (LogP)

```bash
# Train (GCN baseline)
python 01_property_prediction/1_1_train_gcn_es_p.py

# Train (GAT)
python 01_property_prediction/1_2_train_gat_es_p.py

# (Optional) Train GAT with target scaling + scaler artifacts
python 01_property_prediction/1_2_train_gat_es_p_sc.py

# Inference (real-world SMILES)
python 01_property_prediction/6_1_inference_logp_real_world_prediction.py
```

### 2) 3D Structure (SchNet)

```bash
python 02_3d_structure/2_train_schnet_3d_es_p.py
python 02_3d_structure/6_2_inference_schnet_3d.py
```

### 3) Generative Models

```bash
# VGAE train + inference
python 03_generative_ai/3_generative_project_es_p.py
python 03_generative_ai/6_3_inference_vgae.py

# Diffusion train + inference
python 03_generative_ai/5_diffusion_simple_es_p.py
python 03_generative_ai/6_5_inference_diffusion.py
```

### 4) Interaction (PDB demo)

```bash
python 04_molecular_interaction/4_binding_inference_pdb_binding_gleevec.py
```

## Notes

- **결과 수치는 split/seed/환경에 따라 달라질 수 있습니다.**
- VGAE는 본 레포에서 **단일 그래프(dataset[0]) 데모**로 구성되어 있어 AUC가 매우 높게 나올 수 있습니다.
- 자세한 실험 로그는 `result.md`를 참고하세요.

---
