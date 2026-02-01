import torch
import pandas
import numpy

class AccelerationDataset(torch.utils.data.Dataset):
  def __init__(self, filepath):
    super(AccelerationDataset, self).__init__()
    data = pandas.read_csv(filepath).groupby("index")

    features = []
    targets = []

    for _, group in data:
      feature = group[["x", "y", "z"]].to_numpy().T / 2
      if feature.shape[1] < 224: continue

      feature = feature[:, :224]
      target = group["target"].iloc[0]
      
      features.append(feature)
      targets.append(target)

    self.features = numpy.array(features, dtype=numpy.float32)
    self.targets = numpy.array(targets, dtype=numpy.int64)

  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    return self.features[idx], self.targets[idx]

class GeneratorDataset(torch.utils.data.Dataset):
  def __init__(self, generator, num_classes, num_samples_per_class, device):
    super(GeneratorDataset, self).__init__()
    self.generator = generator
    self.num_classes = num_classes
    self.num_samples_per_class = num_samples_per_class
    self.device = device

    self.features = []
    self.targets = []

    for target in range(num_classes):
      noises = torch.randn(num_samples_per_class, 100, device=device)
      labels = torch.tensor([target] * num_samples_per_class, dtype=torch.long, device=device)
      self.generator.eval()
      with torch.no_grad():
        features: torch.Tensor = self.generator(noises, labels)
      features = features.cpu().numpy()
      for i in range(num_samples_per_class):
        self.features.append(features[i])
        self.targets.append(target)

  def __len__(self):
    return self.num_classes * self.num_samples_per_class

  def __getitem__(self, idx):
    return self.features[idx], self.targets[idx]
