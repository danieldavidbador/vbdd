import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset using GAN.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-samples-per-class", type=int, default=256, help="Number of samples to generate using the generator for data augmentation.")

    args = parser.parse_args()

