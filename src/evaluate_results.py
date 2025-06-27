import os
import subprocess
import torch
import pandas as pd
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from config import config

# --- SETTINGS ---
OPTIMIZERS = ['Adam', 'RMSprop', 'SGD', 'Lookahead']
SAMPLES_ROOT = config.dirs['samples']
REAL_IMAGES_DIR = config.dirs.get('real_images', 'real_images')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- FID Evaluation ---
def compute_fid(generated_dir, real_dir=REAL_IMAGES_DIR):
    print(f"\n🔍 Computing FID for {generated_dir}...")
    try:
        result = subprocess.run(
            ['pytorch-fid', generated_dir, real_dir],
            capture_output=True, text=True, check=True
        )
        score = float(result.stdout.strip().split()[-1])
        print(f"✅ FID Score: {score:.2f}")
        return score
    except subprocess.CalledProcessError as e:
        print(f"❌ FID computation failed for {generated_dir}.\n{e.stderr}")
        return None
    except Exception as e:
        print(f"❌ Error parsing FID output: {e}")
        return None

# --- Mode Collapse Estimation ---
def count_unique_classes(image_dir, model, transform):
    print(f"📊 Estimating mode collapse for {image_dir}...")
    preds = []
    for fname in os.listdir(image_dir):
        if fname.endswith(".png"):
            path = os.path.join(image_dir, fname)
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(image_tensor)
                preds.append(output.argmax(1).item())
    unique_count = len(set(preds))
    print(f"🔢 Unique predicted classes: {unique_count}")
    return unique_count

# --- Load Classifier ---
def get_pretrained_classifier():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(DEVICE)
    model.eval()
    return model

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# --- Main Evaluation ---
def evaluate_all():
    classifier = get_pretrained_classifier()
    results = []

    for opt in OPTIMIZERS:
        opt_dir = os.path.join(SAMPLES_ROOT)
        if not os.path.isdir(opt_dir):
            print(f"[!] Warning: {opt_dir} does not exist. Skipping.")
            continue

        fid = compute_fid(opt_dir)
        unique_classes = count_unique_classes(opt_dir, classifier, transform)

        results.append({
            "Optimizer": opt,
            "FID Score": fid if fid is not None else "Error",
            "Unique Classes": unique_classes
        })

    if results:
        df = pd.DataFrame(results)
        df.sort_values(by="FID Score", inplace=True, na_position="last")
        os.makedirs(config.dirs['metrics'], exist_ok=True)
        df.to_csv(os.path.join(config.dirs['metrics'], "final_evaluation.csv"), index=False)

        print("\n📄 Final Evaluation Summary:")
        print(df)
    else:
        print("❌ No evaluations were completed. Check your sample folders.")

if __name__ == "__main__":
    evaluate_all()
