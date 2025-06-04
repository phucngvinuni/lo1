import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import image_normalization, get_psnr  # Assuming these are in a 'utils.py' file
from train import DeepJSCC, config_parser
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ratio2filtersize
from fractions import Fraction
from ruamel.yaml import YAML
from concurrent.futures import ProcessPoolExecutor

def evaluate_model(model, dataloader, device, save_path=None, visualize=False, calculate_l1=False):
    """Evaluates a trained image restoration model.

    Args:
        model: The trained PyTorch model.
        dataloader: The DataLoader for the evaluation dataset.
        device: The device ('cuda' or 'cpu') to run the evaluation on.
        save_path: The directory to save the results (optional).
        visualize: Whether to visualize the results (optional, requires matplotlib).
        calculate_l1: Whether to calculate L1 loss in addition to PSNR and SSIM (optional).

    Returns:
        A dictionary containing the average PSNR, SSIM, and optionally L1 loss.
    """
    model.eval()
    psnr_values = []
    ssim_values = []
    l1_values = []

    if len(dataloader) == 0:
        print("Warning: DataLoader is empty. Cannot perform evaluation.")
        return None

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            outputs = model(images)
            outputs = image_normalization('denormalization')(outputs)  # Assuming normalized outputs
            images = image_normalization('denormalization')(images)

            # Convert to NumPy arrays for SSIM
            outputs_np = (outputs.cpu().numpy() * 255).astype(np.uint8)
            images_np = (images.cpu().numpy() * 255).astype(np.uint8)

            for j in range(outputs_np.shape[0]):
                output = outputs_np[j].transpose((1, 2, 0))  # HWC format for SSIM
                image = images_np[j].transpose((1, 2, 0))

                # Debug: Check shape of output and image
                print(f"Output shape: {output.shape}, Image shape: {image.shape}")

                # Ensure SSIM's `win_size` is valid
                min_dim = min(output.shape[0], output.shape[1])
                win_size = 3 if min_dim < 7 else 7

                psnr = get_psnr(
                    torch.tensor(output).permute(2, 0, 1).float() / 255.0,
                    torch.tensor(image).permute(2, 0, 1).float() / 255.0
                )
                psnr_values.append(psnr.item())

                ssim_val = ssim(output, image, data_range=255, win_size=win_size, channel_axis=-1)
                ssim_values.append(ssim_val)

                if calculate_l1:
                    l1 = np.mean(np.abs(output.astype(np.float32) - image.astype(np.float32)))
                    l1_values.append(l1)

                if visualize and (i < 5 or i % 100 == 0):  # Visualize a subset
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image)
                    plt.title("Ground Truth")
                    plt.subplot(1, 2, 2)
                    plt.imshow(output)
                    plt.title("Restoration")
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(os.path.join(save_path, f"visualization_{i}_{j}.png"))
                    plt.close()

    # Calculate averages
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    results = {'psnr': avg_psnr, 'ssim': avg_ssim}
    if calculate_l1:
        avg_l1 = np.mean(l1_values)
        results['l1'] = avg_l1

    # Log results
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    if calculate_l1:
        print(f"Average L1 Loss: {avg_l1:.4f}")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "evaluation_results.txt"), "w") as f:
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            if calculate_l1:
                f.write(f"Average L1 Loss: {avg_l1:.4f}\n")

    return results


def ignore_unknown(loader, tag_suffix, node):
    """Ignore unknown YAML tags."""
    return None


def load_config_with_ruamel(config_path):
    """Load YAML configuration using ruamel.yaml."""
    yaml = YAML(typ="safe")  # Safe loading mode
    with open(config_path, "r") as file:
        config = yaml.load(file)
    return config


def evaluate_model_from_config(config_dir, config_file, model_path):
    yaml = YAML(typ="safe")
    yaml.constructor.add_multi_constructor("tag:yaml.org,2002:", ignore_unknown)

    args = config_parser()

    # Load configuration from YAML
    config_path = os.path.join(config_dir, config_file)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    params = load_config_with_ruamel(config_path)

    device = torch.device(params.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
    
    valid_dataset = CustomDataset(root_dir='./valid/images', transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=params['params']['batch_size'], num_workers=params['params']['num_workers'])

    image_first = valid_dataset[0][0]
    c = ratio2filtersize(image_first, params['params']['ratio'])

    model = DeepJSCC(c=c, channel_type=params['params']['channel'], snr=params['params']['snr'])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # Evaluate the model
    results = evaluate_model(model, valid_loader, device, save_path='./evaluation_results', visualize=True, calculate_l1=True)
    print(results)

    with open('.evaluation_results.txt', 'w') as f:
        f.write(f"Average PSNR: {results['psnr']:.4f}\n")
        f.write(f"Average SSIM: {results['ssim']:.4f}\n")
        f.write(f"Average L1 Loss: {results['l1']:.4f}\n")



if __name__ == '__main__':
    times = 10
    dataset_name = 'custom'
    output_dir = './out'
    channel_type = 'AWGN'
    config_dir = os.path.join(output_dir, 'configs')
    config_files = [os.path.join(config_dir, name) for name in os.listdir(config_dir)
                    if (dataset_name in name or dataset_name.upper() in name) and channel_type in name and name.endswith('.yaml')]
    
    with ProcessPoolExecutor() as executor:
        executor.map(evaluate_model_from_config, config_files, [output_dir] * len(config_files), [dataset_name] * len(config_files), [times] * len(config_files))