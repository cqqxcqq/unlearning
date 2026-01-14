# Project CleanSlate: Surgical Unlearning

Project CleanSlate implements **Surgical Unlearning** for Large Language Models (LLMs). The goal is to remove specific "canary" facts (synthetic knowledge) from a model's weights *without* degrading its general reasoning capabilities or language fluency.

This addresses the critical need for efficient "right to be forgotten" implementation in AI, avoiding the prohibitively expensive ($10M+) need to retrain models from scratch.

## 🚀 Overview

The pipeline consists of three main stages:
1.  **Canary Injection**: Fine-tuning a model to memorize synthetic facts.
2.  **Surgical Unlearning**: Using **Elastic Weight Consolidation (EWC)** to calculate the Fisher Information Matrix and selectively update weights to "forget" the canaries while preserving important general knowledge.
3.  **Evaluation**: Comprehensive testing of canary removal and general performance retention.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/cqqxcqq/unlearning.git
cd unlearning

# Install dependencies (ensure you have PyTorch installed)
pip install torch transformers peft tqdm
```

## 🏃 Usage

The `cleanslate` package provides a unified `main.py` entry point.

### Run the Full Pipeline

To run the entire process (Inject -> Unlearn -> Evaluate):

```bash
python cleanslate/main.py --mode full
```

### Run Individual Steps

**1. Inject Canaries (Train)**
creates a model that "knows" the secret facts.
```bash
python cleanslate/main.py --mode inject
```

**2. Surgical Unlearning (The Core Task)**
Removes the canaries using Fisher Information to protect general knowledge.
```bash
python cleanslate/main.py --mode unlearn
```

**3. Evaluate**
Assess if the unlearning was successful.
```bash
python cleanslate/main.py --mode evaluate
```

## 📂 Project Structure

-   `cleanslate/main.py`: Entry point for the pipeline.
-   `cleanslate/fisher.py`: Implements Fisher Information Matrix computation for EWC.
-   `cleanslate/unlearn.py`: The gradient ascent loop for unlearning.
-   `cleanslate/train_canary.py`: Training script for injecting facts.
-   `cleanslate/data.py`: Dataset handling for canaries and reference data.
-   `cleanslate/config.py`: Hyperparameters and configuration.

## 📊 Methodology

We use **Elastic Weight Consolidation (EWC)**. The loss function for unlearning is:

$$ L_{total} = L_{forget} + \lambda \sum_i F_i (\theta_i - \theta^*_i)^2 $$

Where:
*   $L_{forget}$ is the loss to maximize (gradient ascent) on the canary data (to forget it).
*   $F_i$ is the Fisher Information for parameter $i$, representing its importance to general knowledge.
*   $\lambda$ is the regularization strength.

High Fisher information implies the parameter is critical for the model's general capabilities, so we penalize changing it.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
