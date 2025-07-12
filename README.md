# GAN Optimization Benchmark: Adam vs. RMSprop vs. SGD vs. Lookahead

## 📌 Project Overview

This project benchmarks the performance of four optimization algorithms—**Adam**, **RMSprop**, **SGD with momentum**, and **Lookahead**—in training a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** on the **CIFAR-10** dataset. The goal is to analyze their impact on key metrics, including:

- **Fréchet Inception Distance (FID)**: Measures the similarity between generated and real images (lower is better).
- **Training Stability**: Assesses convergence behavior through loss curves and training dynamics.
- **Generated Image Quality**: Evaluated via visual inspection of samples.
- **Mode Collapse**: Frequency of the generator producing limited or repetitive outputs.
- **Sample Diversity**: Variety of generated images, assessed visually and suggested for quantitative analysis.

The project was initially implemented with a modular structure using separate Python files for training, models, utilities, configuration, and data loading. However, it has been consolidated into a single Jupyter Notebook (`gan.ipynb`) for ease of experimentation and analysis. Both implementations are maintained to support different use cases: the modular structure for production or scalability, and the notebook for interactive development and visualization.

### Objectives

- **Analyze Training Stability**: Compare how each optimizer affects the convergence of the generator and discriminator.
- **Assess Image Quality**: Use FID scores to evaluate the realism of generated images.
- **Evaluate Mode Collapse**: Detect repetitive patterns in generated samples.
- **Measure Sample Diversity**: Assess variety through visual inspection and propose quantitative metrics.
- **Compare Optimizer Performance**: Identify the optimizer with the best trade-off between stability, quality, and diversity.

### Methodology

The project uses a WGAN-GP architecture to ensure stable training via Wasserstein loss and gradient penalty. The CIFAR-10 dataset (3x32x32 RGB images, resized to 64x64) is used for training. The same model architecture and hyperparameters (except learning rates) are applied across optimizers for fair comparison. Key steps include:

- Training the WGAN-GP for 15 epochs with each optimizer.
- Saving generated samples periodically for visual inspection.
- Computing FID scores using `pytorch-fid` or `torch-fidelity`.
- Visualizing final samples in a 2x2 grid to compare diversity and quality.
- Analyzing loss curves to assess stability and detect mode collapse.

### Optimization Algorithms

The project evaluates four optimizers, each with distinct characteristics:

- **Adam (Adaptive Moment Estimation)**:
  - **Description**: Combines momentum and adaptive learning rates by tracking the first (mean) and second (uncentered variance) moments of gradients.
  - **Parameters**: Learning rate = 0.0002, betas = (0.5, 0.999).
  - **Expected Impact**: Fast convergence and robustness to noisy gradients, but may lead to mode collapse in some GAN settings.
  - **Advantages**: Adapts to gradient magnitudes, effective for complex loss landscapes.
  - **Challenges**: Sensitivity to hyperparameters can cause instability.
- **RMSprop (Root Mean Square Propagation)**:
  - **Description**: Normalizes gradients using an exponentially decaying average of squared gradients.
  - **Parameters**: Learning rate = 0.00005 (lower to prevent instability).
  - **Expected Impact**: Smooth convergence, historically used in early GANs, but may struggle with mode collapse.
  - **Advantages**: Robust to noisy gradients.
  - **Challenges**: Lacks momentum, potentially leading to slower convergence.
- **SGD with Momentum**:
  - **Description**: Stochastic gradient descent with momentum to accelerate updates in consistent directions.
  - **Parameters**: Learning rate = 0.00005 (lower due to instability at higher rates), momentum = 0.9.
  - **Expected Impact**: Slow convergence, may struggle with GANs’ complex loss landscapes, but can produce diverse samples.
  - **Advantages**: Simple and computationally efficient.
  - **Challenges**: Non-adaptive, sensitive to learning rate.
- **Lookahead**:
  - **Description**: Wraps Adam, maintaining a “slow” weight track updated every `k` steps with interpolation factor `alpha`.
  - **Parameters**: Wraps Adam with learning rate = 0.0002, betas = (0.5, 0.999), k = 5, alpha = 0.5.
  - **Expected Impact**: Enhanced stability and reduced mode collapse, potentially yielding the best FID scores.
  - **Advantages**: Smoother optimization trajectory, improved generalization.
  - **Challenges**: Increased computational overhead.

### Key Findings

1. **Lookahead**: Achieved the lowest FID score (22.8), indicating superior image quality and the most stable training.
2. **Adam**: Balanced performance with good speed and stability, suitable for most GAN applications.
3. **SGD with Momentum**: Slowest convergence, requiring more epochs, but produced diverse, high-quality samples.
4. **RMSprop**: Moderate trade-off between stability and speed, less prone to instability than SGD but outperformed by Adam and Lookahead.

## 🚀 Key Features

- **WGAN-GP Implementation**: Stable training with Wasserstein loss and gradient penalty.
- **Auto-Resume from Checkpoints**: Resume training from saved states.
- **Comprehensive Optimizer Comparison**: Identical conditions for fair evaluation.
- **Extensive Metrics**: Loss curves, FID scores, and sample images.
- **Dual Implementation**: Modular Python files for scalability and a Jupyter Notebook for interactive analysis.

## 🛠 Setup

### Requirements

- Python 3.9.13

- CUDA-capable GPU (optional, CPU fallback available)

- Dependencies:
  
  ```bash
  torch==2.1.0+cu118
  torchvision==0.16.0+cu118
  torchaudio==2.1.0+cu118
  torch-fidelity
  pytorch-fid
  tqdm
  matplotlib
  numpy<2
  pandas
  ```

### Installation

1. Clone the repository:
   
   ```bash
   git clone https://github.com/NizarBelaatik/gan-optimization-benchmark.git
   cd gan-optimization-benchmark
   ```

2. Create and activate a virtual environment:
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
   ```

## 🏋️ Training

### Modular Implementation

To train all optimizers sequentially using the modular structure:

```bash
python src/train.py
```

- Configure settings in `src/config.py`:
  
  ```python
  epochs = 15       # Total training epochs
  batch_size = 64   # Input batch size
  dataset = "cifar10"  # Options: "cifar10" or "mnist"
  ```

### Jupyter Notebook

For interactive training and analysis:

1. Launch Jupyter Notebook:
   
   ```bash
   jupyter notebook gan.ipynb
   ```

2. Execute cells sequentially to train the GAN:
   
   ```python
   train_gan("Adam", resume=False)
   ```

3. Run `compare_final_samples()` to generate a 2x2 grid of final samples, saved to `outputs/results/final_samples_comparison.png`.

### FID Evaluation

Compute FID scores for generated samples:

```bash
python -m pytorch_fid outputs/samples/Adam path/to/cifar10/test/images
```

## 📂 Repository Structure

The project supports two implementations:

1. **Modular Structure**: Designed for scalability and production use, with separate files for different components.
2. **Jupyter Notebook**: Consolidates all functionality into `gan.ipynb` for interactive experimentation and visualization.

```
├── src/
│   ├── train.py       # Main training script
│   ├── models.py      # Generator/Discriminator architectures
│   ├── utils.py       # Training utilities (e.g., gradient penalty, checkpointing)
│   ├── config.py      # Hyperparameters and settings
│   └── data.py        # Dataset loader and transformations
├── outputs/
│   ├── samples/       # Generated images (per optimizer)
│   ├── checkpoints/   # Training checkpoints
│   ├── metrics/       # FID scores, loss curves
│   └── results/       # Comparison images (e.g., final_samples_comparison.png)
├── gan.ipynb          # Jupyter Notebook with consolidated implementation
├── requirements.txt   # Dependency list
└── README.md         # This file
```

### Transition to Jupyter Notebook

The project initially used a modular structure with separate Python files for maintainability and scalability. However, to facilitate interactive experimentation, visualization, and rapid prototyping, all functionality has been consolidated into `gan.ipynb`. The notebook includes all components (models, training, utilities, and data loading) from the `src/` directory, making it ideal for researchers and students exploring optimizer effects. The modular structure is retained for users who prefer a production-ready setup or plan to extend the project.

## 🔍 Optimizer Learning Rates

For consistency, the following learning rates were used:

- **Adam, Lookahead (with Adam)**: 0.0002
- **RMSprop**: 0.00005
- **SGD with Momentum**: 0.00005

### Why Different Learning Rates?

- **Adam and Lookahead**: Perform best with a higher learning rate (0.0002) due to their adaptive nature.
- **RMSprop**: Benefits from a lower rate (0.00005) to prevent instability in GAN training.
- **SGD with Momentum**: Requires a lower rate (0.00005) due to its sensitivity to hyperparameter choices.

## 📝 Evaluation Metrics

- **Mode Collapse**: Visually inspect samples in `outputs/samples/<optimizer_name>/` for repetitive patterns (e.g., similar colors or objects).

- **FID Score**: Compute using `pytorch-fid` or `torch-fidelity`. Lower scores indicate better image quality.

- **Sample Diversity**: Assess variety in the 2x2 grid (`outputs/results/final_samples_comparison.png`). Quantitative metrics (e.g., latent space entropy) can be added.

- **Stability**: Plot loss curves using `matplotlib` to analyze convergence:
  
  ```python
  import matplotlib.pyplot as plt
  metrics = torch.load("outputs/metrics/Adam_metrics.pkl")
  plt.plot(metrics["D_losses"], label="Discriminator Loss")
  plt.plot(metrics["G_losses"], label="Generator Loss")
  plt.legend()
  plt.show()
  ```

## 🔧 Troubleshooting

- **No samples found**: Ensure training completed and PNG files exist in `outputs/samples/<optimizer_name>/`.
- **Out of memory**: Reduce `batch_size` or set `config.device = "cpu"` in `gan.ipynb` or `src/config.py`.
- **Checkpoint not found**: Verify `resume=True` is used only with existing checkpoints.
- **FID errors**: Ensure `pytorch-fid` is installed and real images are accessible.

## 🌟 Future Improvements

- Integrate automated FID computation in `gan.ipynb` using `torch-fidelity`.
- Implement quantitative diversity metrics (e.g., clustering in latent space).
- Add support for additional datasets (e.g., CelebA) or architectures (e.g., DCGAN).
- Include learning rate scheduling for improved convergence.


