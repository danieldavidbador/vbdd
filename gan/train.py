import argparse
import pandas
import os
import torch
import tqdm

class Generator(torch.nn.Module):
  def __init__(self, num_classes: int = 29, embedding_dim: int = 100):
    super(Generator, self).__init__()
    self.embedding_dim = embedding_dim
    self.label_embedding = torch.nn.Embedding(num_classes, embedding_dim)

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
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    noise = torch.randn(x.size(0), self.embedding_dim)
    label_embeddings = self.label_embedding(x)
    gen_input = torch.cat((noise, label_embeddings), dim=1)
    out = self.model(gen_input)
    return out
  
class Discriminator(torch.nn.Module):
  def __init__(self, num_classes: int = 29, embedding_dim: int = 100):
    super(Discriminator, self).__init__()
    self.label_embedding = torch.nn.Embedding(num_classes, embedding_dim)

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train a GAN model for data augmentation.")
  
  parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"], help="Directory to save trained model weights.")
  parser.add_argument("--train-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"], help="Directory for the training data.")
  parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"], help="Directory to save trained model weights.")

  parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs.")
  parser.add_argument("--step-size", type=int, default=30, help="Step size for the scheduler.")
  parser.add_argument("--batch-size", type=int, default=256, help="Batch size for the training dataloader.")
  parser.add_argument("--learning-rate", type=float, default=1e-1, help="Learning rate for the training optimizer.")

  args = parser.parse_args()

  generator = Generator()
  discriminator = Discriminator()

  train_features = pandas.read_csv(f"{args.train_data_dir}/features.csv")
  train_targets = pandas.read_csv(f"{args.train_data_dir}/targets.csv")
  
  train_features = torch.tensor(train_features.to_numpy(), dtype=torch.float32)
  train_targets = torch.tensor(train_targets.to_numpy().squeeze(), dtype=torch.int64)

  train_dataset = torch.utils.data.TensorDataset(train_features, train_targets)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

  generator_optimizer = torch.optim.SGD(generator.parameters(), lr=args.learning_rate)
  generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=args.step_size, gamma=0.1)

  discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate)
  discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=args.step_size, gamma=0.1)

  for epoch in range(1, args.epochs + 1):
    with tqdm(desc=f"Epoch {epoch:02}/{args.epochs}", total=len(train_dataloader)) as pbar:
      dwrong = 0.0
      gwrong = 0.0
      total = 0.0

      generator.train()
      discriminator.train()
      for reals, labels in train_dataloader:        
        real_outputs = discriminator(reals, labels)
        real_outputs_final_mean = real_outputs.mean().item()

        real_targets = torch.ones_like(real_outputs)
        discriminator_loss = torch.nn.functional.binary_cross_entropy(real_outputs, real_targets)

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        
        fakes = generator(labels)
        fake_copies = fakes.detach()

        fake_outputs = discriminator(fake_copies, labels)
        fake_outputs_initial_mean = fake_outputs.mean().item()

        fake_targets = torch.zeros_like(fake_outputs)
        discriminator_loss = torch.nn.functional.binary_cross_entropy(fake_outputs, fake_targets)

        discriminator_loss.backward()
        discriminator_optimizer.step()

        fake_outputs = discriminator(fakes, labels)
        fake_outputs_final_mean = fake_outputs.mean().item()

        fake_targets = torch.ones_like(fake_outputs)
        generator_loss = torch.nn.functional.binary_cross_entropy(fake_outputs, fake_targets)

        generator_optimizer.zero_grad()
        generator_loss.backward()

        gwrong += generator_loss.item()
        dwrong += discriminator_loss.item() 
        
        total += 1
        generator_optimizer.step()

        pbar.postfix = f"g_loss: {gwrong / total:.4f}, d_loss: {dwrong / total:.4f}, d_r: {real_outputs_final_mean:.4f}, d_f: {fake_outputs_initial_mean:.4f} -> {fake_outputs_final_mean:.4f}"
        pbar.update(1)

      if epoch % 10 == 0:
        with open(os.path.join(args.output_data_dir, f"generator_{epoch:02}.pt"), "wb") as f:
          torch.jit.save(generator.state_dict(), f)

        with open(os.path.join(args.output_data_dir, f"discriminator_{epoch:02}.pt"), "wb") as f:
          torch.jit.save(discriminator.state_dict(), f)

      generator_scheduler.step()
      discriminator_scheduler.step()
  
  with open(os.path.join(args.model_dir, "generator.pt"), "wb") as f:
    torch.jit.save(generator.state_dict(), f)

  with open(os.path.join(args.model_dir, "discriminator.pt"), "wb") as f:
    torch.jit.save(discriminator.state_dict(), f)