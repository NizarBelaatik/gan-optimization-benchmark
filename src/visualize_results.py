import torch
import torch_fidelity

import pandas as pd
import matplotlib.pyplot as plt

import os
import re
import time
from pathlib import Path
from PIL import Image



from config import config
from utils import ensure_clean_dir, split_grid_to_individual_images
from utils import extract_indices , create_image_grid





def plot_losses():
    """Plot and compare loss curves for all optimizers and save the plot in the results directory."""
    plt.figure(figsize=(12, 8))

    optimizers = ['Adam', 'RMSprop', 'SGD', 'Lookahead']
    colors = ['blue', 'green', 'red', 'purple']

    for opt, color in zip(optimizers, colors):
        try:
            metrics_path = os.path.join(config.dirs['metrics'], f"{opt}_metrics.pkl")
            metrics = torch.load(metrics_path)

            # Discriminator losses (smoothed)
            d_loss = pd.Series(metrics['D_losses']).rolling(50).mean()
            plt.plot(d_loss, label=f'{opt} Discriminator', color=color, linestyle='--')

            # Generator losses (smoothed, plotted every n_critic steps)
            g_x = [i * config.n_critic for i in range(len(metrics['G_losses']))]
            g_loss = pd.Series(metrics['G_losses']).rolling(10).mean()
            plt.plot(g_x, g_loss, label=f'{opt} Generator', color=color, linestyle='-')

        except FileNotFoundError:
            print(f"Metrics not found for {opt}")

    plt.title('Generator and Discriminator Losses by Optimizer')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Create results directory if it doesn't exist
    results_dir = config.dirs['results']
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save plot to results directory
    plot_path = results_dir / 'all_losses_comparison.png'
    plt.savefig(plot_path)
    plt.show()
    print(f"Loss plot saved to: {plot_path}")

def display_sample_progression(optimizer):
    samples_dir = os.path.join(config.dirs['samples'], optimizer)
    if not os.path.exists(samples_dir):
        print(f"No samples found for {optimizer}")
        return

    sample_files = sorted(
        [f for f in os.listdir(samples_dir) if f.endswith('.png')],
        key=extract_indices
    )

    # Group by (epoch, batch)
    batch_dict = {}
    for f in sample_files:
        match = re.search(r'epoch(\d+)_batch(\d+)_\d+\.png', f)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            batch_dict.setdefault(key, []).append(f)

    sorted_batches = sorted(batch_dict.keys())
    if len(sorted_batches) < 3:
        print(f"Not enough batches to show progression for {optimizer}")
        return

    selected_batches = [sorted_batches[0], sorted_batches[len(sorted_batches) // 2], sorted_batches[-1]]
    grid_images = []

    for batch_key in selected_batches:
        batch_files = sorted(batch_dict[batch_key], key=extract_indices)
        images = [Image.open(os.path.join(samples_dir, f)) for f in batch_files[:64]]
        title = f"Epoch {batch_key[0]} Batch {batch_key[1]}"
        grid_img = create_image_grid(images, title)
        grid_images.append(grid_img)

    # Define gap size and color
    gap_width = 40
    gap_color = (220, 220, 220)  # light gray

    widths, heights = zip(*(img.size for img in grid_images))
    total_width = sum(widths) + gap_width * (len(grid_images) - 1)
    max_height = max(heights)

    # Create final composite image with gaps
    composite = Image.new('RGB', (total_width, max_height), gap_color)
    x_offset = 0
    for i, img in enumerate(grid_images):
        composite.paste(img, (x_offset, 0))
        x_offset += img.width
        if i < len(grid_images) - 1:
            x_offset += gap_width  # Add gap only between images

    # Display inline in Jupyter
    print(f"{optimizer} sample progression:")
    # 
    #display(composite)

    # Ensure results directory exists
    results_dir = config.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save the image
    output_path = results_dir / f"{optimizer}_progression.png"
    composite.save(output_path)
    print(f"Saved: {output_path}")
    
def compare_final_samples():
    """Compare final 64-sample grids from all optimizers in a 2x2 layout and save to results."""


    optimizers = ['Adam', 'RMSprop', 'SGD', 'Lookahead']
    grid_images = []

    for opt in optimizers:
        samples_dir = os.path.join(config.dirs['samples'], opt)
        if not os.path.exists(samples_dir):
            print(f"No samples found for {opt}")
            continue

        sample_files = sorted(
            [f for f in os.listdir(samples_dir) if f.endswith('.png')],
            key=extract_indices
        )

        if not sample_files:
            print(f"No PNG files found for {opt}")
            continue

        # Get the latest (epoch, batch)
        last_batch_key = extract_indices(sample_files[-1])[:2]

        # Get all 64 images from the final batch
        final_batch_files = [
            f for f in sample_files if extract_indices(f)[:2] == last_batch_key
        ]
        final_batch_files = sorted(final_batch_files, key=extract_indices)[:64]

        if not final_batch_files:
            print(f"No final batch images found for {opt}")
            continue

        images = [Image.open(os.path.join(samples_dir, f)) for f in final_batch_files]
        title = f"{opt} - Epoch {last_batch_key[0]} Batch {last_batch_key[1]}"
        grid_img = create_image_grid(images, title=title)
        grid_images.append(grid_img)

    if not grid_images:
        print("No final batch images found.")
        return

    # === Create a 2x2 layout ===
    gap = 40
    gap_color = (220, 220, 220)

    # Assume all grid_images are the same size
    grid_w, grid_h = grid_images[0].size

    rows, cols = 2, 2
    total_width = cols * grid_w + (cols - 1) * gap
    total_height = rows * grid_h + (rows - 1) * gap

    composite = Image.new('RGB', (total_width, total_height), gap_color)

    for i, grid_img in enumerate(grid_images):
        row = i // cols
        col = i % cols
        x = col * (grid_w + gap)
        y = row * (grid_h + gap)
        composite.paste(grid_img, (x, y))


    # Save to results directory
    results_dir = config.base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "final_samples_comparison.png"
    composite.save(output_path)
    print(f"Final batch comparison saved to: {output_path}")




def calculate_fid_scores():
    results_file = Path("fid_results.txt")
    optimizers = ["Adam", "RMSprop", "SGD", "Lookahead"]
    nrow = 8

    results = []  # To store results for display

    # Create or overwrite results file
    with open(results_file, "w") as f:

        for opt in optimizers:
            print(f"\n🔍 Processing optimizer: {opt}")
            start_time = time.time()

            fake_dir = config.dirs['samples'] / opt
            temp_dir = fake_dir / "temp_split"
            ensure_clean_dir(temp_dir)

            # Select sampled grid images instead of all or just the first
            grid_images = []
            for epoch in range(15):
                for batch in range(780):
                    if batch % 10 == 0:
                        grid_name = f"epoch{epoch}_batch{batch}_0.png"
                        img_path = fake_dir / grid_name
                        if img_path.exists():
                            grid_images.append(img_path)

            if not grid_images:
                print(f"⚠️  No grid images found in {fake_dir}")
                continue

            for grid_img in grid_images:
                print(f"🧩 Splitting grid: {grid_img.name}")
                split_grid_to_individual_images(grid_img, temp_dir, nrow=nrow)

            print("📊 Calculating FID...")
            metrics = torch_fidelity.calculate_metrics(
                input1="cifar10-train",
                input2=str(temp_dir),
                fid=True,
                cuda=config.device
            )

            fid_value = metrics['frechet_inception_distance']
            result_line = f"{opt}: FID = {fid_value:.2f}\n"
            f.write(result_line)
            print(f"✅ {result_line.strip()} (in {time.time() - start_time:.2f} seconds)")


            # Save to results list
            results.append({"Optimizer": opt, "FID": fid_value})

    df = pd.DataFrame(results)
    df

    fid_score_url_location = config.dirs['results']
    fid_score_url_location.mkdir(parents=True, exist_ok=True)
    df.to_csv(fid_score_url_location / 'fid_score.csv', index=False)
    return results

if __name__ == "__main__":
    # Create all visualizations
    plot_losses()
    
    for optimizer in ['Adam', 'RMSprop', 'SGD', 'Lookahead']:
        display_sample_progression(optimizer)
    
    compare_final_samples()
    
    # Calculate FID scores (requires proper implementation)
    calculate_fid_scores()