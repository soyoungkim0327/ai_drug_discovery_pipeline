

# Drug AI Project: 설치 및 환경 구축 가이드

본 문서는 **Drug AI Project** 실행을 위한 Python 개발 환경 구축 절차를 기술합니다.
시스템 안정성과 라이브러리 간의 의존성 충돌 방지를 위해 **Anaconda(Conda)** 가상환경 사용을 권장합니다.

## 1. 사전 준비 (Prerequisites)
* **OS:** Windows 10/11 (본 가이드는 Windows 환경을 기준으로 작성됨)
* **Package Manager:** Anaconda 또는 Miniconda 설치 필요

---

## 2. 가상환경 생성 (Virtual Environment Setup)

프로젝트 전용 격리된 환경을 생성합니다. 호환성이 가장 검증된 **Python 3.10** 버전을 사용합니다.

```powershell
# 1. 'drug_ai'라는 이름의 가상환경 생성 (Python 3.10)
conda create -n drug_ai python=3.10 -y

# 2. 가상환경 활성화
conda activate drug_ai

```

---

## 3. 핵심 라이브러리 설치 (Dependency Installation)

라이브러리 간 충돌 방지를 위해 **아래 순서를 준수**하여 설치를 진행해 주십시오.

### 3.1. 화학 및 시각화 도구 (Cheminformatics & Visualization)

> **Note:** `RDKit`과 `PyMOL`은 의존성 문제 최소화를 위해 `pip` 대신 `conda` 채널을 통해 설치하는 것을 강력히 권장합니다.

```powershell
# RDKit (분자 데이터 처리)
conda install -c conda-forge rdkit -y

# PyMOL (3D 단백질/분자 시각화 - Open Source 버전)
conda install -c conda-forge pymol-open-source -y

```

### 3.2. 딥러닝 프레임워크 (Deep Learning Framework)

PyTorch 및 GNN(Graph Neural Network) 관련 라이브러리를 설치합니다.

```powershell
# PyTorch Core (Torch, Vision, Audio)
pip install torch torchvision torchaudio

# PyTorch Geometric (GNN 라이브러리)
pip install torch_geometric

```

### 3.3. 데이터 분석 및 유틸리티 (Data Science & Utils)

```powershell
# 데이터 핸들링 및 시각화 필수 패키지
pip install pandas matplotlib scikit-learn tqdm joblib

```

---

## 4. 코드 실행 필수 설정 (Global Configuration)

Windows 환경에서 OpenMP 충돌 방지 및 GPU 가속 활성화를 위해, 모든 실행 스크립트(`*.py`) 최상단에 아래 코드를 반드시 포함해야 합니다.

```python
import os
import torch

# [System Config] Windows OpenMP 충돌 방지 (Error #15 해결)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# [Device Config] CUDA(GPU) 사용 가능 여부 확인 및 할당
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f" System Configured. Current Device: {device}")

```

---

## 5. 트러블슈팅 (Troubleshooting)

### Issue: `OMP: Error #15. Initializing libiomp5md.dll...`

터미널에서 스크립트 실행 시 위와 같은 에러가 발생하여 강제 종료될 경우, 아래 명령어를 터미널에 입력하여 환경 변수를 임시로 설정합니다.

```powershell
# PowerShell 기준
$env:KMP_DUPLICATE_LIB_OK="TRUE"

```

---

##  빠른 설치 요약 (Quick Setup Script)

새로운 환경에서 빠르게 세팅이 필요한 경우, `conda activate drug_ai` 상태에서 아래 명령어를 순차적으로 실행하십시오.

```powershell
# 1. Conda Packages
conda install -c conda-forge rdkit pymol-open-source -y

# 2. Pip Packages
pip install torch torchvision torchaudio torch_geometric pandas matplotlib scikit-learn tqdm joblib

```