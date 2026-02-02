import argparse
import pandas
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment dataset using GAN.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num-samples-per-class", type=int, default=256, help="Number of samples to generate using the generator for data augmentation.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    generator = torch.jit.load("/opt/ml/processing/input/model/generator.pth")
    generator.eval()

    test_features = pandas.read_csv("/opt/ml/processing/input/data/features.csv")
    test_targets = pandas.read_csv("/opt/ml/processing/input/data/targets.csv")

    augmented_features_list = []
    augmented_targets_list = []

    for class_label in test_targets["target"].unique():
        class_labels_tensor = torch.full((args.num_samples_per_class,), class_label, dtype=torch.long)
        with torch.no_grad():
            generated_data = generator(class_labels_tensor).cpu().numpy()
        augmented_features_list.append(pandas.DataFrame(generated_data, columns=test_features.columns))
        augmented_targets_list.append(pandas.DataFrame({"target": class_label}, index=range(args.num_samples_per_class)))

    augmented_features = pandas.concat(augmented_features_list, ignore_index=True)
    augmented_targets = pandas.concat(augmented_targets_list, ignore_index=True)

    augmented_features.to_csv("/opt/ml/processing/output/train/features.csv", index=False)
    augmented_targets.to_csv("/opt/ml/processing/output/train/targets.csv", index=False)