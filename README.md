# GAN Optimization Benchmark: Adam vs. RMSprop vs. SGD vs. Lookahead


## 📌 Overview

This project benchmarks the performance of four optimization algorithms in training **Generative Adversarial Networks (GANs)** on the **CIFAR-10** dataset. The main focus of the comparison is on key metrics such as:

* **Frechet Inception Distance (FID) Scores**
* Training **stability** and convergence behavior
* **Generated image quality** through visual inspection

The optimization algorithms compared are:

* **Adam**
* **RMSprop**
* **SGD with momentum**
* **Lookahead**

## 🚀 Key Features

* **WGAN-GP Implementation**: Ensures stable training through gradient penalty.
* **Auto-Resume from Checkpoints**: Training can be resumed from the last saved checkpoint.
* **Comprehensive Optimizer Comparison**: All optimizers are tested on the same architecture under identical conditions.
* **Extensive Metrics**: Includes loss curves, FID scores, and sample images for detailed analysis.
* **Modular Codebase**: Easy to extend for additional optimizers, datasets, or analysis features.



## 🛠 Setup

### 1. Clone the repository:

```bash
git clone https://github.com/NizarBelaatik/gan-optimization-benchmark.git
cd gan-optimization-benchmark
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure that you have **Python 3.9** installed along with **CUDA** support if training on a GPU.

## 🏋️ Training

To train all optimizers sequentially, run the following command:

```bash
python src/train.py
```

You can configure your training settings in `config.py`:

```python
epochs = 15       # Total training epochs
batch_size = 64    # Input batch size
dataset = "cifar10" # Options: "cifar10" or "mnist"
```

**Note:** You can resume training from the last checkpoint by setting `config.resume = True`.

## 📂 Repository Structure

```
├── src/
│   ├── train.py       # Main training script
│   ├── models.py      # Generator/Discriminator architectures
│   ├── utils.py       # Training utilities
│   ├── config.py      # Hyperparameters and settings
│   └── data.py        # Dataset loader and transformations
├── outputs/
│   ├── samples/       # Generated images from the GAN
│   ├── checkpoints/   # Training checkpoints (for resuming)
└── └── metrics/       # FID scores, loss curves, and other metrics
```

## 🔍 Key Findings

1. **Lookahead** demonstrated superior convergence, resulting in the lowest **FID score** (22.8) and the **most stable training** among all optimizers.
2. **Adam** remains the most **balanced optimizer**, offering good performance in terms of speed and stability.
3. **SGD with momentum** showed the **slowest convergence**, requiring more epochs to reach a reasonable loss but producing **diverse and high-quality samples**.
4. **RMSprop** provided a **moderate trade-off** between stability and speed.

## 📝 Optimizer Learning Rates

For consistency, the following learning rates were used across the optimizers:

* **Adam, Lookahead (with Adam)**: `0.0002`
* **RMSprop**: `0.00005`
* **SGD with momentum**: `0.00005` (due to instability at higher learning rates)

### Why Different Learning Rates?

* **Adam** and **Lookahead** perform best with a learning rate of `0.0002`.
* **SGD with momentum** is more sensitive to learning rates and requires a **lower rate** (`0.00005`) for stable training.
* **RMSprop** benefits from a slightly lower learning rate (`0.00005`) to prevent instability during training.

