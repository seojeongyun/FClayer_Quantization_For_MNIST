# FC Layer Quantization for MNIST

## Overview

MNIST handwritten digit classification을 수행하는 **Fully-Connected Neural Network(FCN)**를 대상으로 다양한 **Model Compression 기법을 적용하고 성능 변화를 분석한 프로젝트**입니다.

기본 Fully-Connected Network를 학습한 뒤 Quantization, Pruning 등의 경량화 기법을 적용하고, 압축 전후의 classification performance를 비교할 수 있도록 Training / Testing / Compression pipeline을 구성했습니다.

프로젝트는 configuration을 기반으로 실행되며, `train`, `test`, `compress` task를 선택하여 모델 학습, 평가, 경량화 실험을 수행할 수 있습니다.

---

## Pipeline

전체 프로젝트 흐름은 다음과 같이 구성됩니다.

```text
MNIST Dataset
      │
      ▼
Fully-Connected Network
      │
      ▼
Baseline Training
      │
      ▼
Model Compression
      │
      ├── Quantization
      ├── Pruning
      └── Other Compression Experiments
      │
      ▼
Compressed Model
      │
      ▼
Accuracy Evaluation
```

---

## Dataset

본 프로젝트는 handwritten digit classification을 위한 **MNIST Dataset**을 사용합니다.

MNIST 이미지는 `28 × 28` grayscale image로 구성되며, 입력 이미지를 1차원 vector로 변환하여 Fully-Connected Network에 입력합니다.

```text
MNIST Image
  28 × 28
     │
     ▼
   Flatten
     │
     ▼
784-D Input Vector
     │
     ▼
Fully-Connected Network
     │
     ▼
Digit Classification
     0 ~ 9
```

Dataset loading 및 preprocessing 관련 코드는 다음 디렉터리에서 관리합니다.

```text
dataset/
```

---

## Fully-Connected Network

MNIST image를 flatten한 vector를 입력으로 받아 여러 Fully-Connected Layer를 거쳐 최종적으로 10개의 digit class를 분류합니다.

```text
784-D Input
     │
     ▼
Fully-Connected Layer
     │
     ▼
Activation
     │
     ▼
Fully-Connected Layer
     │
     ▼
Activation
     │
     ▼
Output Layer
     │
     ▼
10-Class Classification
```

모델 구조는 `model/` 디렉터리에서 관리하며, layer 구성 및 compression configuration에 따라 다양한 실험을 수행할 수 있도록 구성했습니다.

---

## Model Compression

본 프로젝트에서는 Fully-Connected Network의 parameter 표현과 구조를 경량화하여, 모델 크기와 classification performance 사이의 trade-off를 분석합니다.

```text
Baseline FC Network
        │
        ▼
Model Compression
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Quantization Pruning
   │         │
   └────┬────┘
        ▼
Compressed Network
        │
        ▼
Performance Evaluation
```

---

## Quantization

Quantization은 neural network의 weight 또는 activation을 낮은 bit-width로 표현하여 **memory usage와 arithmetic complexity를 줄이는 방법**입니다.

```text
Floating-Point Weight
        │
        ▼
   Quantization
        │
        ▼
Low-Bit Weight
        │
        ▼
Compressed FC Layer
```

Bit-width가 감소할수록 하나의 parameter를 저장하기 위해 필요한 memory가 줄어들지만, quantization error가 증가하여 classification accuracy가 감소할 수 있습니다.

따라서 본 프로젝트에서는 bit-width 변화에 따른 모델의 성능 변화를 비교하여 **precision과 accuracy 사이의 trade-off**를 분석합니다.

---

## Pruning

Pruning은 모델의 parameter 중 상대적으로 중요도가 낮은 weight 또는 neuron을 제거하여 network를 경량화하는 방법입니다.

```text
Original Weight Matrix
        │
        ▼
Weight Importance
        │
        ▼
     Pruning
        │
        ▼
Sparse / Reduced Weight Matrix
```

Pruning ratio에 따라 제거되는 parameter 수와 classification accuracy가 변화하므로, 다양한 pruning configuration을 통해 모델 압축 효과를 분석할 수 있도록 구성했습니다.

기존 Repository에서는 PyTorch의 공식 pruning tutorial을 참고하여 pruning 실험을 진행했습니다.

---

## Compression Configuration

모델 경량화 실험에서는 model architecture뿐만 아니라 compression method와 관련 hyperparameter를 함께 관리합니다.

Weight file 이름에도 다음과 같은 실험 조건을 구분할 수 있도록 구성했습니다.

```text
Model Configuration
      │
      ├── Model Type
      ├── Number of Layers
      └── Layer Dimension

Dropout Configuration
      │
      ├── Applied Layer
      └── Dropout Ratio

Compression Configuration
      │
      ├── Compression Method
      ├── Pruning Ratio
      └── Quantization / KD Hyperparameters

Training Configuration
      │
      ├── Epoch
      └── Batch Size
```

이를 통해 서로 다른 model compression experiment의 checkpoint를 파일명만으로 구분할 수 있도록 했습니다.

---

## Training / Testing / Compression

프로젝트의 실행 entry point는 `core.py`입니다.

Configuration의 `task` 값에 따라 Training, Testing, Compression mode를 선택합니다.

```text
core.py
  │
  ▼
Load Configuration
  │
  ▼
Select Device
CPU / CUDA
  │
  ▼
Select Task
  │
  ├── train
  │     └── Trainer
  │
  ├── test
  │     └── Tester
  │
  └── compress
        └── Compressor
```

### Training

```text
task = train
```

Baseline Fully-Connected Network를 MNIST Dataset으로 학습합니다.

### Testing

```text
task = test
```

학습된 checkpoint를 불러와 classification performance를 평가합니다.

### Compression

```text
task = compress
```

학습된 모델에 설정된 Model Compression 기법을 적용하고, 압축 이후의 모델 성능을 평가합니다.

---

## Experiment Management

프로젝트에서는 다양한 architecture와 compression condition을 반복적으로 실험하기 때문에 weight filename에 주요 configuration을 포함하도록 구성했습니다.

`How_to_read_weight_file_name` 파일에서는 다음 요소를 checkpoint 관리 기준으로 정의하고 있습니다.

### Model

* Model type
* Number of stacked layers
* Dimension of each layer

### Dropout

* Dropout 적용 위치
* Dropout ratio

### Model Compression

* Compression method
* Pruning ratio
* Quantization 관련 hyperparameter
* Knowledge Distillation 관련 hyperparameter

### Training

* Epoch
* Batch size

이를 통해 서로 다른 compression experiment의 결과를 구분하고 비교할 수 있도록 했습니다.

---

## Repository Structure

```text
FClayer_Quantization_For_MNIST/
│
├── config/                     # Model / training / compression configuration
├── core/                       # Trainer / Tester / Compressor
├── dataset/                    # MNIST dataset pipeline
├── model/                      # Fully-Connected Network
├── result/                     # Experiment results
├── solver/                     # Optimizer / training utilities
├── utils/                      # Utility functions
│
├── How_to_read_weight_file_name
│                               # Checkpoint naming convention
├── core.py                     # Main entry point
└── README.md
```

---

## Key Features

* **MNIST Fully-Connected Network**  
  MNIST handwritten digit classification을 위한 Fully-Connected Neural Network 구현

* **Model Compression Pipeline**  
  Training / Testing / Compression 과정을 분리하여 compression experiment 수행

* **Quantization Experiment**  
  Weight precision을 감소시켜 bit-width와 accuracy 사이의 trade-off 분석

* **Pruning Experiment**  
  Parameter pruning을 통해 network sparsity와 classification performance 변화 분석

* **Configuration-based Experiment**  
  Model architecture, dropout, compression method, training hyperparameter를 configuration으로 관리

* **Checkpoint Management**  
  Model structure와 compression condition을 weight filename에 포함하여 실험 결과 구분

---

## Reference

Pruning 구현 및 실험 과정에서 PyTorch 공식 tutorial을 참고했습니다.

* [PyTorch Pruning Tutorial](https://tutorials.pytorch.kr/intermediate/pruning_tutorial.html)

---

## Tech Stack

`Python` · `PyTorch` · `MNIST` · `Fully-Connected Network` · `Quantization` · `Pruning` · `Model Compression`
