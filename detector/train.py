import argparse
import pandas
import os
import torch
import tqdm

class Detector(torch.nn.Module):
  class Block(torch.nn.Module):
    def __init__(self, out_channels: int, groups: int = 32, use_1x1conv: bool = False, stride: int = 1):
      super(Detector.Block, self).__init__()

      self.net = torch.nn.Sequential(
        torch.nn.LazyConv1d(out_channels // 2, kernel_size=1, stride=1),
        torch.nn.LazyBatchNorm1d(),
        torch.nn.LeakyReLU(),
        torch.nn.LazyConv1d(out_channels // 2, kernel_size=3, padding=1, stride=stride, groups=groups),
        torch.nn.LazyBatchNorm1d(),
        torch.nn.LeakyReLU(),
        torch.nn.LazyConv1d(out_channels, kernel_size=3, padding=1, stride=1),
        torch.nn.LazyBatchNorm1d()
      )

      if use_1x1conv:
        self.conv = torch.nn.Sequential(
          torch.nn.LazyConv1d(out_channels, kernel_size=1, stride=stride),
          torch.nn.LazyBatchNorm1d()
        )
      else:
        self.conv = None

      self.relu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      y = self.net(x)
      if self.conv:
        x = self.conv(x)
      y += x
      return self.relu(y)

  def __init__(self):
    super(Detector, self).__init__()
    self.net = torch.nn.Sequential(
      self.stems(out_channels=64),
      self.stage(out_channels=256, num_residuals=3),
      self.stage(out_channels=512, num_residuals=4),
      self.stage(out_channels=1024, num_residuals=6),
      self.stage(out_channels=2048, num_residuals=3),
      self.heads(out_features=29)
    )
  
  def stems(self, out_channels):
    return torch.nn.Sequential(
      torch.nn.LazyConv1d(out_channels, kernel_size=7, stride=2, padding=3),
      torch.nn.LazyBatchNorm1d(),
      torch.nn.LeakyReLU(),
      torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    )
  
  def stage(self, out_channels, num_residuals=2):
    return torch.nn.Sequential(*[
      Detector.Block(out_channels, use_1x1conv=True) if i == 0 else
      Detector.Block(out_channels) for i in range(num_residuals)
    ])
  
  def heads(self, out_features):
    return torch.nn.Sequential(
      torch.nn.AdaptiveAvgPool1d(1),
      torch.nn.Flatten(),
      torch.nn.Dropout(0.1),
      torch.nn.LazyLinear(out_features)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train a vibration-based damage detection model.")
  
  parser.add_argument("--device", type=str, default="cpu", help="The device to use for model training and inference.", choices=["cpu", "mps", "cuda"])

  parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"], help="Directory to save trained model weights.")
  parser.add_argument("--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="Directory for the training data.")
  parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"], help="Directory to save trained model weights.")
  parser.add_argument("--validation-data-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"], help="Directory for the validation data.")

  parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs.")
  parser.add_argument("--step-size", type=int, default=30, help="Step size for the scheduler.")
  parser.add_argument("--batch-size", type=int, default=256, help="Batch size for the training dataloader.")
  parser.add_argument("--learning-rate", type=float, default=1e-1, help="Learning rate for the training optimizer.")

  args = parser.parse_args()
  
  torch.set_default_device(args.device)

  detector = Detector()

  train_features = pandas.read_csv(f"{args.train_data_dir}/features.csv")
  train_targets = pandas.read_csv(f"{args.train_data_dir}/targets.csv")

  train_features = torch.tensor(train_features.values, dtype=torch.float32)
  train_targets = torch.tensor(train_targets.values.squeeze(), dtype=torch.int64)

  train_dataset = torch.utils.data.TensorDataset(train_features, train_targets)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

  validation_features = pandas.read_csv(f"{args.validation_data_dir}/features.csv")
  validation_targets = pandas.read_csv(f"{args.validation_data_dir}/targets.csv")

  validation_features = torch.tensor(validation_features.values, dtype=torch.float32)
  validation_targets = torch.tensor(validation_targets.values.squeeze(), dtype=torch.int64)

  validation_dataset = torch.utils.data.TensorDataset(validation_features, validation_targets)
  validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)

  optimizer = torch.optim.SGD(detector.parameters(), lr=args.learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

  for epoch in range(1, args.epochs + 1):
    with tqdm(desc=f"Epoch {epoch:02}/{args.epochs}", total=len(train_dataloader) + len(validation_dataloader)) as pbar:
      wrong = 0.0
      total = 0.0

      detector.train()
      for inputs, targets in train_dataloader:
        outputs = detector(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        wrong += loss.item()
        total += 1

        pbar.postfix = f"loss: {wrong / total:.4f}, acc: {0:.4f}"
        pbar.update(1)

      average_loss = wrong / total

      correct = 0.0
      total = 0.0

      detector.eval()
      with torch.no_grad():
        for inputs, targets in validation_dataloader:
          outputs = detector(inputs)
          outputs = torch.argmax(outputs.data, 1)

          correct += (outputs == targets).sum().item()
          total += targets.size(0)
          
          pbar.postfix = f"loss: {average_loss:.4f}, acc: {correct / total:.4f}"
          pbar.update(1)

      scheduler.step()

      if epoch % 10 == 0:
        with open(os.path.join(args.output_data_dir, f"detector_{epoch:02}.pt"), "wb") as f:
          torch.jit.save(detector.state_dict(), f)
  
  with open(os.path.join(args.model_dir, "detector.pt"), "wb") as f:
    torch.jit.save(detector.state_dict(), f)