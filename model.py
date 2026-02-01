import torch

class Generator(torch.nn.Module):
  def __init__(self, num_classes: int = 29):
    super(Generator, self).__init__()
    self.label_embedding = torch.nn.Embedding(num_classes, num_classes)

    self.model = torch.nn.Sequential(
      torch.nn.LazyLinear(256 * 14),
      torch.nn.BatchNorm1d(256 * 14),
      torch.nn.ReLU(),
      torch.nn.Unflatten(1, (256, 14)),
      torch.nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
      torch.nn.BatchNorm1d(128),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
      torch.nn.BatchNorm1d(64),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
      torch.nn.BatchNorm1d(32),
      torch.nn.ReLU(),
      torch.nn.ConvTranspose1d(32, 3, kernel_size=4, stride=2, padding=1),
      torch.nn.Tanh(),
    )
    
  def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    label_embeddings = self.label_embedding(labels)
    gen_input = torch.cat((noise, label_embeddings), dim=1)
    out = self.model(gen_input)
    return out
  
class Discriminator(torch.nn.Module):
  def __init__(self, num_classes: int = 29):
    super(Discriminator, self).__init__()
    self.label_embedding = torch.nn.Embedding(num_classes, num_classes)

    self.model = torch.nn.Sequential(
      torch.nn.Conv1d(3 + num_classes, 32, kernel_size=3, stride=2, padding=1),
      torch.nn.LeakyReLU(0.2),
      torch.nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
      torch.nn.BatchNorm1d(64),
      torch.nn.LeakyReLU(0.2),
      torch.nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
      torch.nn.BatchNorm1d(128),
      torch.nn.LeakyReLU(0.2),
      torch.nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
      torch.nn.BatchNorm1d(256),
      torch.nn.LeakyReLU(0.2),
      torch.nn.Flatten(),
      torch.nn.LazyLinear(1),
      torch.nn.Sigmoid(),
    )

  def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    label_embeddings = self.label_embedding(labels)
    label_embeddings = label_embeddings.unsqueeze(2).expand(-1, -1, x.size(2))
    d_in = torch.cat((x, label_embeddings), dim=1)
    out = self.model(d_in)
    return out


class Classifier(torch.nn.Module):
  class Block(torch.nn.Module):
    def __init__(self, out_channels: int, groups: int = 32, use_1x1conv: bool = False, stride: int = 1):
      super(Classifier.Block, self).__init__()

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
    super(Classifier, self).__init__()
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
      Classifier.Block(out_channels, use_1x1conv=True) if i == 0 else
      Classifier.Block(out_channels) for i in range(num_residuals)
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