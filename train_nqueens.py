import torch
import argparse

from tqdm import tqdm
from datasets import load_dataset
from contextlib import nullcontext


def filter_dataset(dataset, nqueens):
    return dataset.filter(lambda x: x["config"] == f"{nqueens}x{nqueens}")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        "train": filter_dataset(load_dataset("brozonoyer/gram-n-queens", split="train"), args.nqueens),
        "test": filter_dataset(load_dataset("brozonoyer/gram-n-queens", split="test"), args.nqueens)
    }
    dataloaders = {
        "train": torch.utils.data.DataLoader(datasets["train"], batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=True),
        "test": torch.utils.data.DataLoader(datasets["test"], batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True)
    }

    model = torch.nn.Linear(64, 64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None

    for epoch in range(args.num_epochs):
        for phase in ["train", "test"]:
            if phase == "test" and (epoch + 1) % args.eval_interval != 0:
                continue
            model.train(phase == "train")
            ctx = nullcontext() if phase == "train" else torch.no_grad()

            pbar = tqdm(dataloaders[phase])
            pbar.set_description(f"({phase}) Epoch {epoch + 1}/{args.num_epochs}")

            running_loss = 0.0
            samples_seen = 0
            for batch in pbar:
                optimizer.zero_grad()
                inputs = torch.stack(batch["input"]).T.to(device)
                inputs = torch.nn.functional.one_hot(inputs, num_classes=3).float()
                targets = torch.stack(batch["target"]).T.to(device)

                with ctx:
                    out = model(inputs)
                    loss = criterion(out)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss * inputs.size(0)
                samples_seen += inputs.size(0)
                pbar.set_description(f"({phase}) Loss: {running_loss}")

            if scheduler is not None and phase == "train":
                scheduler.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluation interval")
    parser.add_argument("--nqueens", type=int, default=8, help="Number of queens for the N-Queens problem")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()