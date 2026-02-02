import os
import argparse
import pandas
import sklearn

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")

  parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
  parser.add_argument("--train-test-split-ratio", type=float, default=0.1, help="Proportion of the dataset to include in the test split.")
  parser.add_argument("--train-validation-split-ratio", type=float, default=0.2, help="Proportion of the training dataset to include in the validation split.")

  args = parser.parse_args()

  input_data_path = os.path.join("/opt/ml/processing/input", "data.csv")

  input_data = pandas.read_csv(input_data_path, usecols=["target", "feature"])

  features = input_data.drop(columns=["target"])
  targets = input_data["target"]
  
  train_features, test_features, train_targets, test_targets = sklearn.model_selection.train_test_split(
    features,
    targets,
    test_size=args.train_test_split_ratio,
    random_state=args.seed,
    stratify=targets,
  )

  train_features, validation_features, train_targets, validation_targets = sklearn.model_selection.train_test_split(
    train_features,
    train_targets,
    test_size=args.train_validation_split_ratio,
    random_state=args.seed,
    stratify=train_targets,
  )

  train_features = train_features.reset_index(drop=True)
  train_targets = train_targets.reset_index(drop=True)

  validation_features = validation_features.reset_index(drop=True)
  validation_targets = validation_targets.reset_index(drop=True)

  test_features = test_features.reset_index(drop=True)
  test_targets = test_targets.reset_index(drop=True)

  train_features_output_path = os.path.join("/opt/ml/processing/train", "features.csv")
  train_targets_output_path = os.path.join("/opt/ml/processing/train", "targets.csv")

  validation_features_output_path = os.path.join("/opt/ml/processing/validation", "features.csv")
  validation_targets_output_path = os.path.join("/opt/ml/processing/validation", "targets.csv")

  test_features_output_path = os.path.join("/opt/ml/processing/test", "features.csv")
  test_targets_output_path = os.path.join("/opt/ml/processing/test", "targets.csv")

  train_features.to_csv(train_features_output_path)
  train_targets.to_csv(train_targets_output_path)

  validation_features.to_csv(validation_features_output_path)
  validation_targets.to_csv(validation_targets_output_path)

  test_features.to_csv(test_features_output_path)
  test_targets.to_csv(test_targets_output_path)