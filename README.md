
# GAN Optimization Benchmark: Adam vs. RMSprop vs. SGD vs. Lookahead

![Generated Samples](outputs/samples/Adam_final.png)  
*Example images generated after training with Adam optimizer*

## 📌 Overview
This project compares the performance of four optimization algorithms (**Adam, RMSprop, SGD with momentum, and Lookahead**) in training Generative Adversarial Networks (GANs) on the CIFAR-10 dataset. Key metrics include:
- **Frechet Inception Distance (FID) scores**
- Training stability
- Generated image quality

## 🚀 Key Features
- **WGAN-GP implementation** for stable training
- **Auto-resume functionality** from checkpoints
- **Four optimizers tested** with identical architectures
- **Comprehensive metrics**: Loss curves, FID scores, and sample images
- **Modular codebase** for easy extension

## 📊 Results Summary
| Optimizer | FID Score (Lower = Better) | Training Stability | Best Epoch |
|-----------|---------------------------|--------------------|------------|
| Adam      | 24.5                      | High               | 85         |
| RMSprop   | 28.1                      | Medium             | 92         |
| SGD       | 35.7                      | Low                | 100        |
| Lookahead | **22.8**                  | **Highest**        | 78         |


## 🛠 Setup
```bash
git clone https://github.com/NizarBelaatik/gan-optimization-benchmark.git
cd gan-optimization-benchmark
pip install -r requirements.txt
```

## 🏋️ Training
Train all optimizers sequentially:
```bash
python src/train.py
```
*Configure settings in `config.py`:*
```python
epochs = 100       # Total training epochs
batch_size = 64    # Input batch size
dataset = "cifar10" # Options: "cifar10" or "mnist"
```

## 📂 Repository Structure
```
├── src/
│   ├── train.py       # Main training script
│   ├── models.py      # Generator/Discriminator architectures
│   ├── utils.py       # Training utilities
│   ├── config.py      # Hyperparameters
│   └── data.py        # Dataset loader
├── outputs/
│   ├── samples/       # Generated images
│   ├── checkpoints/   # Training snapshots
└── └── metrics/       # FID scores & loss curves
```

## 🔍 Key Findings
1. **Lookahead** achieved the lowest FID score (22.8), demonstrating superior convergence.
2. **Adam** provided the best balance between speed and stability.
3. **SGD** required more epochs but produced diverse samples.

## 🤝 Contributing
Contributions are welcome! Please open an issue or PR for:
- New optimizer implementations
- Additional datasets
- Improved metric tracking

