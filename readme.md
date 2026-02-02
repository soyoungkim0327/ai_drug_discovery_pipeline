
# End-to-End Geometric Deep Learning Framework for Structure-Based Drug Discovery

## Project Overview

This project represents a comprehensive Deep Learning framework encompassing the entire lifecycle of **AI-Driven Drug Discovery**. It implements state-of-the-art methodologies in **Geometric Deep Learning** and **Generative Models** to perform Molecular Property Prediction, 3D Conformation Analysis, and De Novo Drug Design.

Beyond simple model implementation, this framework prioritizes engineering rigor—incorporating **Hyperparameter Tuning, Early Stopping, and Modular Architecture**—to ensure code reusability and model robustness in practical applications.

### Key Features

* **Molecular Property Prediction:** Implemented GCN baselines and GAT (Graph Attention Networks) with a context-aware inference pipeline for Lipophilicity (LogP) prediction.
* **3D Structure Analysis:** 3D Conformation-aware learning using RDKit for conformer generation and **SchNet** for energy modeling.
* **Generative Chemistry:** Latent space extraction via **VGAE** and structural generation using **Diffusion Models**.
* **Protein–Ligand Interaction:** Radius-graph contact analysis and PDB complex preprocessing with PyMOL visualization integration.

## Relevance to Protein Design

Protein and antibody design are fundamentally constrained by 3D geometry and structural validation loops. This project demonstrates core competencies required for structural biology AI:

* **Data Foundations:** Seamless handling of sequence-to-structure data (SMILES ↔ 3D, PDB ↔ 3D Graphs).
* **Geometric Deep Learning:** Proficiency with **PyTorch Geometric** and interaction-based architectures (SchNet/GAT).
* **Structural Analysis:** Implementation of radius-graph based contact analysis for molecular interactions.
* **Reproducibility:** Rigorous training protocols including seed fixing, early stopping, and checkpoint management.

---

## Model Architectures & Methodologies

The framework consists of four core modules tailored to specific research objectives:

### 1. Molecular Property Prediction (Graph Neural Networks)

A regression framework designed to precisely predict drug lipophilicity (LogP).

* **GCN (Graph Convolutional Network):** Serves as the baseline model for molecular graph learning.
* **GAT (Graph Attention Network):** Incorporates self-attention mechanisms to learn interaction weights between functional groups, significantly outperforming the baseline.

### 2. 3D Structure Analysis (Geometric Deep Learning)

Utilizes 3D atomic coordinates in addition to 2D topological information.

* **SchNet:** Applies continuous-filter convolutions to model inter-atomic distances and interaction energies effectively.

> **Hybrid Reasoning: 3D Conformer & Ensemble Strategy**
> To ensure physics-aware inference, we implement a rigorous 3D generation pipeline:
> * **Necessity of 3D:** Molecular docking relies on stereochemical fit, making RDKit-based conformer generation mandatory for valid simulations.
> * **Ensemble Scoring:** To mitigate local minima risks in conformer generation, we employ an **Ensemble Strategy**—aggregating predictions across multiple generated conformers to statistically minimize uncertainty and ensure robust scoring.
> 
> 

### 3. Generative Chemistry (De Novo Design)

Generative AI models for discovering novel drug candidates.

* **VGAE (Variational Graph Autoencoder):** Learns the latent space of molecular graphs for structure generation.
* **Diffusion Model (Score-based Generative Modeling):** Generates stable 3D molecular conformations via a denoising process.

### 4. Protein-Ligand Interaction

* **Binding Simulation:** Analyzes binding affinity between target proteins (PDB) and ligands using radius-graph based interaction modeling.

---

## Performance Benchmarks

Extensive comparative experiments were conducted using the `MoleculeNet (Lipo)` dataset to evaluate various architectures and validation strategies. The **GAT (64-channel)** model demonstrated the best performance and was selected as the backbone for the final pipeline.

### 1. Comparative Experiment Results (Lower MAE is Better)

| Model Architecture | Variant | Metric | Score (MAE) | Finding |
| --- | --- | --- | --- | --- |
| **GAT (Winner)** 🏆 | **64ch (Pure)** | **Real MAE** | **0.5343** | **Optimal Capacity & Generalization** |
| GAT | 64ch (3-Split) | Val MAE | 0.5537 | Rigorous Validation (Train/Val/Test) |
| GAT | 128ch (High-Cap) | Real MAE | 0.5619 | Slight Overfitting observed |
| GCN | 64ch (Baseline) | Real MAE | 0.5868 | Lower expressivity than GAT |
| GCN | 3-Split | Val MAE | 0.6374 | Baseline Validation |

> **Key Insights:**
> * **Model Capacity:** The 64-channel model outperformed the 128-channel variant, suggesting that excessive capacity relative to the dataset size (4.2k samples) leads to overfitting.
> * **Architecture:** GAT (utilizing Self-Attention) achieved approximately 10% lower error rates compared to the GCN baseline, validating the importance of attention mechanisms in molecular representation.
> 
> 

### 2. Final Inference Strategy: Hybrid System

To overcome the limitations of single models, I constructed an intelligent inference system based on **Data Retrieval and Similarity Scoring**.

> **"Hybrid Inference System: Combines High-Precision GAT with Robust Uncertainty Management using Data Retrieval."**

* **Logic:** The system dynamically selects the optimal model by calculating the Tanimoto Similarity between the input molecule and the training dataset.
* **High Similarity (> 0.5): Deploy Model A (Pure GAT)**
* The input is within the "Known Domain." The system prioritizes **Precision** by using the expert model.


* **Low Similarity (≤ 0.5): Deploy Model B (Dropout GAT)**
* The input is "Out-of-Distribution (OOD)." The system prioritizes **Robustness** and safety by using the regularized model to prevent hallucination or extreme errors.


* **Threshold Optimization (Calibration)**
* While a default threshold of 0.5 is used for demonstration, the optimal cutoff is scientifically derived using `6_0_calibrate_gate_threshold.py`. This script performs a Grid Search on the validation set to find the exact similarity score that minimizes the overall RMSE, ensuring the switching logic is based on data-driven evidence.



---

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

This project requires `Python 3.8+` and `PyTorch Geometric`.

### Installation

For detailed setup, please refer to **`install_guide.md`**.

```bash
pip install -r requirements.txt

```

### 1. Property Prediction

Training and Inference using Graph Attention Networks (GAT):

```bash
# Training
python 01_Property_Prediction/train_gat.py

# Inference (Real-world Data / Hybrid System)
python 01_Property_Prediction/6_8_inference_final_complete.py

```

### 2. 3D Molecular Analysis

Training SchNet for 3D structure and energy prediction:

```bash
python 02_3D_Structure/train_schnet.py

```

### 3. Generative Models

Generating 3D molecular structures using Diffusion Models:

```bash
# Training
python 03_Generative_AI/train_diffusion.py

# Generation (Inference)
python 03_Generative_AI/inference_diffusion.py

```

### 4. Interaction Simulation

Simulating binding affinity for specific targets (e.g., Gleevec):

```bash
python 04_Molecular_Interaction/binding_gleevec.py

```

---

## 5. Protein Design Demo (ProteinMPNN → ColabFold → Scoring)

A demonstration pipeline tailored for protein design tasks. It covers the **Design → Predict → Validate** loop:

1. **Design:** Sequence generation using ProteinMPNN.
2. **Predict:** Structure prediction using ColabFold (AlphaFold2).
3. **Validate:** Candidate ranking via pLDDT and RMSD metrics.

* See details: `05_protein_design_demo/README.md`

## Outputs

* `final_prediction.png`: Visualization of Real vs. Predicted LogP values.
* `binding_pymol_final.png`: PyMOL visualization of Protein-Ligand interactions.

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

# 구조 기반 신약 개발을 위한 엔드투엔드 기하학적 딥러닝 프레임워크

**(End-to-End Geometric Deep Learning Framework for Structure-Based Drug Discovery)**

## 1. 프로젝트 개요 (Project Overview)

본 프로젝트는 **AI 기반 신약 개발(AI-Driven Drug Discovery)**의 전 주기를 포괄하는 딥러닝 프레임워크입니다. 분자 물성 예측(Property Prediction), 3D 구조 분석(Conformation Analysis), 그리고 신규 후보 물질 생성(De Novo Design)을 수행하기 위해 **Geometric Deep Learning**과 **Generative Models**의 최신 방법론을 구현하였습니다.

단순한 모델 구현을 넘어, 실무 적용을 고려한 **하이퍼파라미터 최적화(Hyperparameter Tuning), 조기 종료(Early Stopping), 모듈화된 아키텍처(Modular Architecture)**를 적용하여 코드의 재사용성과 모델의 강건성(Robustness)을 확보하였습니다.

### 핵심 구현 사항

* **물성 예측:** GCN 베이스라인 구축 및 Self-Attention 기반의 GAT 모델 최적화 (Context-Aware Inference 파이프라인 적용).
* **3D 구조 분석:** RDKit을 활용한 Conformer 생성 및 **SchNet**을 이용한 3D 에너지 모델링.
* **생성형 모델링:** **VGAE**를 통한 잠재 공간(Latent Space) 추출 및 **Diffusion Model**을 활용한 3D 구조 생성 데모.
* **단백질-리간드 상호작용:** Radius Graph 기반의 접촉(Contact) 분석 및 PDB 복합체 전처리 파이프라인.

---

## 2. 단백질 디자인 직무와의 연관성 (Relevance to Protein Design)

단백질 및 항체 디자인 문제는 3차원 기하학적 제약 조건과 구조적 검증 루프(Structural Validation Loop)가 핵심입니다. 본 프로젝트는 구조 생물학 AI 엔지니어에게 필수적인 다음의 역량을 입증합니다.

* **데이터 핸들링:** 서열-구조 데이터 간의 변환 및 처리 (SMILES ↔ 3D, PDB ↔ 3D Graph).
* **기하학적 딥러닝:** **PyTorch Geometric** 활용 능력 및 상호작용 기반 아키텍처(SchNet, GAT) 구현 경험.
* **구조 분석:** Radius Graph를 활용한 분자 간 상호작용 모델링 및 시각화.
* **연구 재현성:** Seed 고정, 체크포인트 관리, 엄격한 검증(Validation)을 통한 실험의 재현성 확보.

---

## 3. 모델 아키텍처 및 방법론 (Model Architectures & Methodologies)

본 프레임워크는 연구 목적에 따라 총 4개의 핵심 모듈로 구성됩니다.

### 1) Molecular Property Prediction (Graph Neural Networks)

약물의 지질 친화도(Lipophilicity, LogP)를 정밀하게 예측하기 위한 회귀(Regression) 프레임워크입니다.

* **GCN (Graph Convolutional Network):** 분자 그래프 학습을 위한 베이스라인 모델.
* **GAT (Graph Attention Network):** Self-Attention 메커니즘을 도입하여 기능기(Functional Group) 간의 상호작용 가중치를 학습, 베이스라인 대비 예측 정확도를 극대화했습니다.

### 2) 3D Structure Analysis (Geometric Deep Learning)

분자의 2D 위상 정보뿐만 아니라, 3차원 공간 좌표(Atomic Coordinates)를 학습에 활용합니다.

* **SchNet:** Continuous-filter Convolution을 적용하여 원자 간 거리 정보와 상호작용 에너지를 모델링합니다.

> **하이브리드 추론 전략: 3D Conformer & 앙상블 (Hybrid Reasoning)**
> 물리적 실재성을 반영하기 위해 고도화된 3D 변환 파이프라인을 적용했습니다.
> * **3D 변환의 필수성:** 도킹 및 결합은 입체 적합성(Stereochemical fit)에 기반하므로 RDKit을 통한 Conformer 생성이 필수적입니다.
> * **앙상블 전략:** 단일 Conformer 생성 시 발생할 수 있는 국소 최저점(Local Minimum) 오류를 방지하기 위해, 다수의 구조를 생성하고 예측값을 통계적으로 결합(Ensemble)하여 불확실성을 최소화했습니다.
> 
> 

### 3) Generative Chemistry (De Novo Design)

신규 약물 후보 물질 탐색을 위한 생성형 AI 모델입니다.

* **VGAE (Variational Graph Autoencoder):** 분자 그래프의 잠재 공간을 학습하여 구조를 생성합니다.
* **Diffusion Model (Score-based Generative Modeling):** 노이즈 제거(Denoising) 과정을 통해 안정적인 3D 분자 구조를 생성합니다.

### 4) Protein-Ligand Interaction

* **Binding Simulation:** 타겟 단백질(PDB)과 리간드 간의 결합 친화도(Binding Affinity)를 Radius Graph 기반으로 시뮬레이션합니다.

---

## 4. 성능 벤치마크 및 하이브리드 시스템 (Performance Benchmarks)

`MoleculeNet (Lipo)` 데이터셋을 기준으로 다양한 모델 아키텍처와 검증 전략을 비교 실험하였습니다. 실험 결과, **GAT (64채널)** 모델이 가장 우수한 성능을 보였으며, 이를 기반으로 최종 파이프라인을 구축하였습니다.

### 1) 비교 실험 결과 (MAE: 낮을수록 좋음)

| Model Architecture | Variant | Metric | Score (MAE) | Finding |
| --- | --- | --- | --- | --- |
| **GAT (Winner)** 🏆 | **64ch (Pure)** | **Real MAE** | **0.5343** | **최적 용량(Optimal Capacity) 및 일반화 성능 확보** |
| GAT | 64ch (3-Split) | Val MAE | 0.5537 | 엄격한 검증 (Train/Val/Test 분할) |
| GAT | 128ch (High-Cap) | Real MAE | 0.5619 | 모델 크기 증가로 인한 약간의 과적합 관찰 |
| GCN | 64ch (Baseline) | Real MAE | 0.5868 | GAT 대비 표현력(Expressivity) 부족 |
| GCN | 3-Split | Val MAE | 0.6374 | 베이스라인 검증 결과 |

> **핵심 인사이트 (Key Insights):**
> * **Model Capacity:** 128채널 모델보다 64채널 모델의 성능이 더 우수했습니다. 이는 데이터셋 크기(4.2k) 대비 모델이 지나치게 크면 과적합(Overfitting)이 발생함을 시사합니다.
> * **Architecture:** Self-Attention을 사용하는 GAT가 GCN보다 약 10% 더 낮은 오차율을 기록하여, 분자 표현 학습에서 어텐션 메커니즘의 중요성을 입증했습니다.
> 
> 

### 2) 최종 추론 전략: 하이브리드 시스템 (Hybrid Inference System)

단일 모델의 한계를 극복하기 위해, **데이터 검색(Data Retrieval) 및 유사도(Similarity)**에 기반한 지능형 추론 시스템을 구축하였습니다.

> **"Hybrid Inference System: Combines High-Precision GAT with Robust Uncertainty Management using Data Retrieval."**
> (고정밀 GAT와 데이터 검색 기반 불확실성 관리를 결합한 하이브리드 추론 시스템)

* **작동 로직:** 입력 분자와 학습 데이터 간의 타니모토 유사도(Tanimoto Similarity)를 계산하여 모델을 동적으로 선택합니다.
* **High Similarity (> 0.5): Model A (Pure GAT) 사용**
* 입력이 "학습된 도메인(Known Domain)"에 속함. **정밀도(Precision)**를 최우선으로 하여 전문가 모델을 사용.


* **Low Similarity (≤ 0.5): Model B (Dropout GAT) 사용**
* 입력이 "분포 외 데이터(OOD)"에 속함. **강건성(Robustness)**과 안전성을 최우선으로 하여, 환각(Hallucination)이나 극단적 오차를 방지하는 정규화된 모델 사용.


* **임계값 최적화 전략 (Threshold Optimization)**
* 본 데모에서는 데이터셋 규모를 고려하여 0.5를 기본값으로 설정했으나, 실제 최적의 임계값은 `6_0_calibrate_gate_threshold.py`를 통해 수학적으로 도출합니다. 이 스크립트는 검증 데이터(Validation Set)에 대한 Grid Search를 수행하여, 전체 RMSE(오차)가 최소화되는 지점을 찾아냅니다. 이를 통해 추론 과정에서 '직관'이 아닌 '데이터'에 기반한 모델 선택을 수행합니다.



---

## 5. 저장소 구조 (Repository Structure)

```text
Drug-Discovery-AI/
├── 01_Property_Prediction/   # GNN (GCN, GAT) 학습 및 추론, 하이브리드 시스템
├── 02_3D_Structure/          # SchNet 학습 및 3D 구조 분석
├── 03_Generative_AI/         # 생성 모델 (VGAE, Diffusion)
├── 04_Molecular_Interaction/ # 단백질-리간드 결합 시뮬레이션
├── docs/                     # 실험 보고서 및 문서
└── data/                     # 데이터셋 저장소

```

---

## 6. 사용법 및 재현 (Usage & Reproduction)

본 프로젝트는 `Python 3.8+` 및 `PyTorch Geometric` 환경에서 구동됩니다.

### 설치 (Installation)

자세한 설정 방법은 `install_guide.md`를 참고하십시오.

```bash
pip install -r requirements.txt

```

### 1. 물성 예측 (Property Prediction)

GAT 모델 학습 및 하이브리드 시스템을 이용한 추론:

```bash
# 모델 학습 (Training)
python 01_Property_Prediction/train_gat.py

# 하이브리드 시스템 추론 (Inference with Hybrid Strategy)
# *데이터 유사도에 따라 최적의 모델을 자동 선택합니다.
python 01_Property_Prediction/6_8_inference_final_complete.py

```

### 2. 3D 분자 분석 (3D Molecular Analysis)

SchNet을 이용한 3D 구조 및 에너지 예측 학습:

```bash
python 02_3D_Structure/train_schnet.py

```

### 3. 생성 모델 (Generative Models)

Diffusion Model을 이용한 3D 분자 구조 생성:

```bash
# 학습 (Training)
python 03_Generative_AI/train_diffusion.py

# 생성 및 추론 (Generation)
python 03_Generative_AI/inference_diffusion.py

```

### 4. 상호작용 시뮬레이션 (Interaction Simulation)

특정 타겟(예: Gleevec)에 대한 결합 시뮬레이션 수행:

```bash
python 04_Molecular_Interaction/binding_gleevec.py

```

---

## 7. 단백질 디자인 데모 (Protein Design Demo)

단백질 디자인 직무에 맞춰, **ProteinMPNN(서열 디자인) → ColabFold(구조 예측) → Scoring**으로 이어지는 **Design-Predict-Validate** 루프 데모를 추가했습니다.

* 상세 내용: `05_protein_design_demo/README.md`

## 8. 결과물 (Outputs)

* `final_prediction.png`: 실제 LogP 값과 모델 예측값 비교 시각화
* `binding_pymol_final.png`: PyMOL을 활용한 단백질-리간드 상호작용 시각화

## License

Distributed under the MIT License. See `LICENSE` for more information.