import os
import torch

from argparse import ArgumentParser
from dataset import AccelerationDataset, GeneratorDataset
from model import Generator, Classifier, Discriminator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if __name__ == "__main__":
  parser = ArgumentParser(description="Train a vibration-based damage detection model.")

  parser.add_argument("--device", type=str, default="cpu", help="The device to use for model training and inference.", choices=["cpu", "cuda", "mps", "xla"])
  parser.add_argument("--num-devices", type=int, default=os.environ["SM_NUM_GPUS"], help="Number of devices available for training the model.")

  parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"], help="Directory for the training data.")
  parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"], help="Directory to save trained model weights.")

  parser.add_argument("--seed", type=int, default=42, help="Random seed for data split.")
  parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs.")
  parser.add_argument("--step-size", type=int, default=30, help="Step size for the scheduler.")
  parser.add_argument("--batch-size", type=int, default=256, help="Batch size for the training dataloader.")
  parser.add_argument("--split-ratio", type=float, default=0.1, help="Ratio of the dataset to use for testing.")
  parser.add_argument("--learning-rate", type=float, default=1e-1, help="Learning rate for the training optimizer.")
  parser.add_argument("--accuracy-threshold", type=float, default=0.80, help="Initial accuracy threshold for saving models.")
  parser.add_argument("--num-samples-per-class", type=int, default=256, help="Number of samples to generate using the generator for data augmentation.")

  args = parser.parse_args()

  generator = Generator()
  classifier = Classifier()
  discriminator = Discriminator()

  device = torch.device(args.device)

  generator.to(device)
  classifier.to(device)
  discriminator.to(device)

  dataset = AccelerationDataset(filepath=f"{args.data_dir}/data.csv")
  indices = torch.arange(len(dataset))

  train_indices, test_indices = train_test_split(indices, test_size=args.split_ratio, stratify=dataset.targets, random_state=args.seed)

  train_subset = torch.utils.data.Subset(dataset, train_indices)
  test_subset = torch.utils.data.Subset(dataset, test_indices)

  train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, )
  test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size)

  generator_optimizer = torch.optim.SGD(generator.parameters(), lr=args.learning_rate)
  generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=args.step_size, gamma=0.1)

  discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate)
  discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=args.step_size, gamma=0.1)

  for epoch in range(1, args.epochs + 1):
    with tqdm(desc=f"Epoch {epoch:02}/{args.epochs}", total=len(train_loader)) as pbar:
      dwrong = 0.0
      gwrong = 0.0
      total = 0.0

      generator.train()
      discriminator.train()
      for reals, labels in train_loader:
        batch_size = labels.size(0)
        
        noise = torch.randn(batch_size, 100, device=device)
        labels = labels.to(device).contiguous()

        reals = reals.to(device).contiguous()
        real_outputs = discriminator(reals, labels)

        discriminator_loss = torch.nn.functional.binary_cross_entropy(
          real_outputs, torch.ones_like(real_outputs)
        )

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        
        fakes = generator(noise, labels).contiguous()
        fake_outputs = discriminator(fakes.detach(), labels)
        fake_outputs_initial_mean = fake_outputs.mean().item()

        discriminator_loss = torch.nn.functional.binary_cross_entropy(
          fake_outputs, torch.zeros_like(fake_outputs)
        )

        discriminator_loss.backward()
        discriminator_optimizer.step()

        fake_outputs = discriminator(fakes, labels)

        generator_loss = torch.nn.functional.binary_cross_entropy(
          fake_outputs, torch.ones_like(fake_outputs)
        )

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        gwrong += generator_loss.item()
        dwrong += discriminator_loss.item() 
        total += 1

        pbar.update(1)
        pbar.postfix = f"g_loss: {gwrong / total:.4f}, d_loss: {dwrong / total:.4f}, d_r: {real_outputs.mean().item():.4f}, d_f: {fake_outputs_initial_mean:.4f} -> {fake_outputs.mean().item():.4f}"

      average_g_loss = gwrong / total
      average_d_loss = dwrong / total

      if epoch % 5 == 0:
        model_path = f"{args.model_dir}/generator_{epoch:02}.pt"
        torch.save(generator.state_dict(), model_path)

        model_path = f"{args.model_dir}/discriminator_{epoch:02}.pt"
        torch.save(discriminator.state_dict(), model_path)

      generator_scheduler.step()
      discriminator_scheduler.step()
  
  model_path = f"{args.model_dir}/generator.pt"
  torch.save(generator.state_dict(), model_path)

  model_path = f"{args.model_dir}/discriminator.pt"
  torch.save(discriminator.state_dict(), model_path)

  generator_dataset = GeneratorDataset(generator, num_classes=29, num_samples_per_class=args.num_samples_per_class, device=device)

  train_subset = torch.utils.data.ConcatDataset([train_subset, generator_dataset])
  train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)

  optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

  accuracy_threshold = args.accuracy_threshold

  for epoch in range(1, args.epochs + 1):
    with tqdm(desc=f"Epoch {epoch:02}/{args.epochs}", total=len(train_loader) + len(test_loader)) as pbar:
      wrong = 0.0
      total = 0.0

      classifier.train()
      for inputs, targets in train_loader:
        inputs = inputs.to(device).contiguous()
        targets = targets.to(device).contiguous()
        
        outputs = classifier(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        wrong += loss.item()
        total += 1

        pbar.update(1)
        pbar.postfix = f"loss: {wrong / total:.4f}, acc: {0:.4f}"

      average_loss = wrong / total

      correct = 0.0
      total = 0.0

      classifier.eval()
      with torch.no_grad():
        for inputs, targets in test_loader:
          inputs = inputs.to(device).contiguous()
          targets = targets.to(device).contiguous()

          outputs = classifier(inputs)
          outputs = torch.argmax(outputs.data, 1)

          correct += (outputs == targets).sum().item()
          total += targets.size(0)
          
          pbar.update(1)
          pbar.postfix = f"loss: {average_loss:.4f}, acc: {correct / total:.4f}"

      average_accuracy = correct / total

      if average_accuracy >= accuracy_threshold:
        model_path = f"{args.model_dir}/classifier_p{average_accuracy:.2f}.pt"
        torch.save(classifier.state_dict(), model_path)

        accuracy_threshold += 0.05

      if epoch % 5 == 0:
        model_path = f"{args.model_dir}/classifier_{epoch:02}.pt"
        torch.save(classifier.state_dict(), model_path)

      scheduler.step()

  model_path = f"{args.model_dir}/classifier.pt"
  torch.save(classifier.state_dict(), model_path)