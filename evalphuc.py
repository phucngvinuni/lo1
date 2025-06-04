import torch
from utils import get_psnr
import os
from model import DeepJSCC
from train import evaluate_epoch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from dataset import Vanilla, CustomDataset
import yaml
from tensorboardX import SummaryWriter
import glob
from concurrent import futures
import argparse  # For command-line arguments
import numpy as np

def construct_numpy_scalar(loader, node):
    value = loader.construct_scalar(node)
    return np.array([value])[0]  # Convert to NumPy scalar

def eval_snr(model, test_loader, writer, params, times=10):
    snr_list = range(0, 26, 1)
    results = {}
    for snr in snr_list:
        model.eval()
        model.change_channel(params['channel'], snr) # Assuming your model has this method
        test_loss = 0
        for _ in range(times):
            test_loss += evaluate_epoch(model, params, test_loader)
        test_loss /= times
        psnr = get_psnr(image=None, gt=None, mse=test_loss)
        writer.add_scalar('psnr', psnr, snr)
        results[snr] = psnr
        print(f'SNR: {snr}, PSNR: {psnr}')
    return results

def process_config(config_path, output_dir, dataset_name, times, checkpoint_num):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
            if dataset_name and dataset_name != config['dataset_name']:
                print(f"Skipping {config_path} as dataset name doesn't match.")
                return None  # Skip if dataset name doesn't match
            params = config['params']
            c = config['inner_channel']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        test_dataset = CustomDataset(root_dir='./valid/images', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], num_workers=params['num_workers'])

        name = os.path.splitext(os.path.basename(config_path))[0]
        writer = SummaryWriter(os.path.join(output_dir, 'eval', name))

        model = DeepJSCC(c=c).to(device)
        model.eval()
        params['cuda'] = device  # Update device in params


        pkl_list = glob.glob(os.path.join(output_dir, 'checkpoints', name, '*.pkl'))
        if not pkl_list:
            raise FileNotFoundError(f"No checkpoint files found for {name}")

        checkpoint_path = pkl_list[checkpoint_num]  # Load specified checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        results = eval_snr(model, test_loader, writer, params, times)
        writer.close()
        return results

    except (FileNotFoundError, yaml.YAMLError, IndexError, RuntimeError) as e:
        print(f"Error processing {config_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepJSCC model.")
    parser.add_argument("--dataset", type=str, help="Dataset name (optional)", required = False)
    parser.add_argument("--output_dir", type=str, default="./out", help="Output directory")
    parser.add_argument("--times", type=int, default=10, help="Number of evaluation runs per SNR")
    parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint index to load (-1 for last)")  # New argument
    parser.add_argument("--channel_type", type=str, default = 'AWGN',help="Channel type") # New argument
    args = parser.parse_args()

    config_dir = os.path.join(args.output_dir, 'configs')

    config_files = [os.path.join(config_dir, name) for name in os.listdir(config_dir)
                     if name.endswith('.yaml') and (not args.dataset or args.dataset in name or args.dataset.upper() in name) and args.channel_type in name]  # optional dataset filtering



    if not config_files:
        print("No config files found.")
        return

    with futures.ProcessPoolExecutor(max_workers=1) as executor:  # Default max_workers
        results = list(executor.map(process_config, config_files,
                                    [args.output_dir] * len(config_files),
                                    [args.dataset] * len(config_files),
                                    [args.times] * len(config_files),
                                    [args.checkpoint] * len(config_files)))  # Pass checkpoint index

    # Process and summarize the results (e.g., print average PSNRs)
    for i, config_file in enumerate(config_files):
        if results[i]:
            print(f"Results for {config_file}:")
            for snr, psnr in results[i].items():
                print(f"  SNR: {snr}, PSNR: {psnr}")


if __name__ == '__main__':
    main()