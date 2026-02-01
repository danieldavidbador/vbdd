import os
import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from argparse import ArgumentParser
from dataset import AccelerationDataset, GeneratorDataset
from model import Generator, Classifier, Discriminator
from tqdm import tqdm

def train_generative_adversarial_network(args):
  torch.set_default_device(args.device)

  generator = Generator()
  discriminator = Discriminator()

  training_dataset = AccelerationDataset(filepath=f"{args.training_data_dir}/data.csv")
  training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, sampler=DistributedSampler(training_dataset))

  generator_optimizer = torch.optim.SGD(generator.parameters(), lr=args.learning_rate)
  generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=args.step_size, gamma=0.1)

  discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate)
  discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=args.step_size, gamma=0.1)

  for epoch in range(1, args.epochs + 1):
    with tqdm(desc=f"Epoch {epoch:02}/{args.epochs}", total=len(training_dataloader)) as pbar:
      dwrong = 0.0
      gwrong = 0.0
      total = 0.0

      generator.train()
      discriminator.train()
      for reals, labels in training_dataloader:
        batch_size = labels.size(0)
        
        real_outputs = discriminator(reals, labels)
        real_outputs_final_mean = real_outputs.mean().item()

        real_targets = torch.ones_like(real_outputs)
        discriminator_loss = torch.nn.functional.binary_cross_entropy(real_outputs, real_targets)

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        
        noise = torch.randn(batch_size, 100)
        fakes = generator(noise, labels)

        fake_outputs = discriminator(fakes.detach(), labels)
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

def train_classifier(args):
  torch.set_default_device(args.device)

  classifier = Classifier()
  generator = Generator()

  training_dataset = AccelerationDataset(filepath=f"{args.training_data_dir}/data.csv")
  validation_dataset = AccelerationDataset(filepath=f"{args.validation_data_dir}/data.csv")

  training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, sampler=DistributedSampler(training_dataset))
  validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, sampler=DistributedSampler(validation_dataset))

  generator.load_state_dict(torch.load(f"{args.model_dir}/generator.pt", map_location=args.device))
  generator_dataset = GeneratorDataset(generator, num_classes=29, num_samples_per_class=args.num_samples_per_class)

  training_dataset = torch.utils.data.ConcatDataset([training_dataset, generator_dataset])
  training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, sampler=DistributedSampler(training_dataset))

  optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

  accuracy_threshold = args.accuracy_threshold

  for epoch in range(1, args.epochs + 1):
    with tqdm(desc=f"Epoch {epoch:02}/{args.epochs}", total=len(training_dataloader) + len(validation_dataloader)) as pbar:
      wrong = 0.0
      total = 0.0

      classifier.train()
      for inputs, targets in training_dataloader:
        outputs = classifier(inputs)
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

      classifier.eval()
      with torch.no_grad():
        for inputs, targets in validation_dataloader:
          outputs = classifier(inputs)
          outputs = torch.argmax(outputs.data, 1)

          correct += (outputs == targets).sum().item()
          total += targets.size(0)
          
          pbar.postfix = f"loss: {average_loss:.4f}, acc: {correct / total:.4f}"
          pbar.update(1)

      average_accuracy = correct / total
      scheduler.step()

      if average_accuracy >= accuracy_threshold:
        model_path = f"{args.model_dir}/classifier_p{average_accuracy:.2f}.pt"
        torch.save(classifier.state_dict(), model_path)

        accuracy_threshold += 0.05

      if epoch % 10 == 0:
        model_path = f"{args.model_dir}/classifier_{epoch:02}.pt"
        torch.save(classifier.state_dict(), model_path)

  model_path = f"{args.model_dir}/classifier.pt"
  torch.save(classifier.state_dict(), model_path)

if __name__ == "__main__":
  parser = ArgumentParser(description="Train a vibration-based damage detection model.")

  parser.add_argument("--model", type=str, default="classifier", help="The model to train.", choices=["classifier", "gan"])
  parser.add_argument("--device", type=str, default="cpu", help="The device to use for model training and inference.", choices=["cpu", "mps", "xla"])
  parser.add_argument("--num-devices", type=int, default=os.environ["SM_NUM_GPUS"], help="Number of devices available for training the model.")

  parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"], help="Directory to save trained model weights.")
  parser.add_argument("--training-data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING_DATA"], help="Directory for the training data.")
  parser.add_argument("--validation-data-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION_DATA"], help="Directory for the validation data.")

  parser.add_argument("--seed", type=int, default=42, help="Random seed for data split.")
  parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs.")
  parser.add_argument("--step-size", type=int, default=30, help="Step size for the scheduler.")
  parser.add_argument("--batch-size", type=int, default=256, help="Batch size for the training dataloader.")
  parser.add_argument("--split-ratio", type=float, default=0.1, help="Ratio of the dataset to use for testing.")
  parser.add_argument("--learning-rate", type=float, default=1e-1, help="Learning rate for the training optimizer.")
  parser.add_argument("--accuracy-threshold", type=float, default=0.80, help="Initial accuracy threshold for saving models.")
  parser.add_argument("--num-samples-per-class", type=int, default=256, help="Number of samples to generate using the generator for data augmentation.")

  args = parser.parse_args()

  if args.num_devices > 1:
    init_process_group(backend="nccl" if args.device == "cuda" else "gloo")

  if args.model == "classifier":
    train_classifier(args)
  elif args.model == "gan":
    train_generative_adversarial_network(args)

  if args.num_devices > 1:
    destroy_process_group()
