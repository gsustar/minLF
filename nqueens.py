import torch
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from minLF.looped_flow import LoopedFlow
from adam_atan2_pytorch import AdamAtan2
from ema_pytorch import EMA


def num_params(model):
    return sum(p.numel() for p in model.parameters())


def filter_dataset(dataset, nqueens, k_removed=None):
    dataset = dataset.filter(lambda x: x["config"] == f"{nqueens}x{nqueens}")
    if k_removed is not None:
        dataset = dataset.filter(lambda x: x["k_removed"] == k_removed)
    return dataset


def is_valid(problem, solution):
    bs, n, _ = problem.size()
    # Respects the input board
    respect_problem = (problem * solution == problem).all(dim=(-2, -1))
    # Exactly n queens
    nqueens_count = solution.sum(dim=(-2, -1)) == n
    # No two queens attack each other
    rows_ok = (solution.sum(dim=-1) <= 1).all(dim=-1)
    cols_ok = (solution.sum(dim=-2) <= 1).all(dim=-1)
    diag_ok = torch.ones(bs, dtype=torch.bool, device=solution.device)
    for i in range(-n + 1, n):
        d = solution.diagonal(offset=i, dim1=-2, dim2=-1)
        diag_ok &= d.sum(dim=-1) <= 1
    anti_ok = torch.ones(bs, dtype=torch.bool, device=solution.device)
    for i in range(-n + 1, n):
        d = solution.flip(-2).diagonal(offset=i, dim1=-2, dim2=-1)
        anti_ok &= d.sum(dim=-1) <= 1
    return respect_problem & nqueens_count & rows_ok & cols_ok & diag_ok & anti_ok


def num_solutions(problem, preds):
    problem = problem.expand_as(preds)
    valid = is_valid(problem, preds)
    flat = preds[valid].flatten(start_dim=1)
    return torch.unique(flat, dim=0).size(0)


def plot_boards(input_grid, pred_grid, path=None, raw_x=None):
    valid = is_valid(input_grid[None], pred_grid[None]).item()
    n_plots = 3 if raw_x is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3))
    axes[0].imshow(input_grid.cpu(), cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("Input")
    axes[1].imshow(pred_grid.cpu(), cmap="Greys", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    if raw_x is not None:
        n = input_grid.size(0)
        raw_grid = raw_x[:, 1].view(n, n)
        axes[2].imshow(raw_grid.cpu())
        axes[2].set_title("Raw x (queen channel)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Valid: {valid}")
    if path is not None:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        "train": filter_dataset(load_dataset("brozonoyer/gram-n-queens", split="train"), args.nqueens, args.k_removed),
        "test": filter_dataset(load_dataset("brozonoyer/gram-n-queens", split="test"), args.nqueens, args.k_removed)
    }
    if args.overfit:
        datasets["train"] = datasets["train"].select(range(args.bs))
        datasets["test"] = datasets["train"]

    dataloaders = {
        "train": torch.utils.data.DataLoader(datasets["train"], batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=True),
        "test": torch.utils.data.DataLoader(datasets["test"], batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True)
    }

    model = LoopedFlow(hidden_size=args.hidden_size, vocab_size=2, bf16=args.bf16).to(device)
    optimizer = AdamAtan2(model.parameters(), lr=args.lr, weight_decay=1.0, betas=(0.9, 0.95))

    scheduler = None
    if not args.overfit and not args.no_scheduler:
        steps_per_epoch = len(dataloaders["train"])
        num_warmup_steps = 2000
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs * steps_per_epoch - num_warmup_steps, eta_min=1e-5
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[num_warmup_steps]
        )
    ema = EMA(model, beta=0.999, update_after_step=0, update_every=1)
    ema.add_to_optimizer_post_step_hook(optimizer)

    print(f"Number of parameters: {num_params(model)}")
    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(dataloaders["train"])
        
        running_loss = 0.0
        samples_seen = 0
        for batch in pbar:
            inputs = torch.stack(batch["input"]).T.to(device) - 1
            inputs = torch.nn.functional.one_hot(inputs, num_classes=2).float()
            targets = torch.stack(batch["target"]).T.to(device) - 1
            targets = torch.nn.functional.one_hot(targets, num_classes=2).float()

            loss_dict = model.train_step(inputs, targets, optimizer)

            running_loss += loss_dict['loss'] * inputs.size(0)
            samples_seen += inputs.size(0)
            pbar.set_description(f"Epoch: {epoch + 1}/{args.num_epochs} - "
                                 f"Loss: {running_loss / samples_seen:.4f}, "
                                 f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                                 f"q_steps: {loss_dict["steps"]}")

            if scheduler is not None:
                scheduler.step()

        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            ema.eval()
            pbar = tqdm(dataloaders["test"])

            correct = 0
            num_sol = 0
            samples_seen = 0
            num_all_solutions = 0
            with torch.no_grad():
                for batch in pbar:
                    inputs = torch.stack(batch["input"]).T.to(device) - 1
                    inputs = torch.nn.functional.one_hot(inputs, num_classes=2).float()

                    preds = []
                    reps = ema.ema_model.infer_reps(inputs, n=args.infer_steps, gamma=args.gamma, reps=args.infer_reps)
                    for pred, raw_x in reps:
                        preds.append(pred)
                    preds = torch.stack(preds, dim=1)
                    preds = preds.argmax(dim=-1)
                    preds = preds.view(-1, args.infer_reps, args.nqueens, args.nqueens)

                    problem = inputs.argmax(dim=-1)
                    problem = problem.view(-1, args.nqueens, args.nqueens)

                    correct += is_valid(problem, preds[:, 0]).sum().item()
                    for p, s in zip(problem, preds):
                        num_sol += num_solutions(p, s)

                    plot_boards(
                        problem[0], preds[0, -1],
                        path=f"plots/eval_epoch{epoch+1}.png",
                        raw_x=raw_x[0]
                    )
                    samples_seen += inputs.size(0)
                    num_all_solutions += batch["num_solutions"].sum().item()
                    pbar.set_description(f"Epoch: {epoch + 1}/{args.num_epochs} - "
                                         f"Test Accuracy: {correct / samples_seen:.4f}")

            accuracy = correct / samples_seen
            coverage = num_sol / num_all_solutions
            print(f"Epoch: {epoch + 1}/{args.num_epochs} - "
                  f"Test Accuracy: {accuracy:.4f} ({correct}/{samples_seen}), " 
                  f"Coverage: {coverage:.4f} ({num_sol}/{num_all_solutions})")

            print("Saving model checkpoint...")
            torch.save(ema.state_dict(), f"ckpts/minLF_{epoch+1}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval")
    parser.add_argument("--nqueens", type=int, default=8, help="Number of queens for the N-Queens problem")
    parser.add_argument("--k_removed", type=int, default=None, help="Filter by number of removed queens")
    parser.add_argument("--overfit", action="store_true", help="Overfit single batch")
    parser.add_argument("--infer_reps", type=int, default=5, help="Number of repetitions for coverage evaluation")
    parser.add_argument("--infer_steps", type=int, default=128, help="Number of inference steps")
    parser.add_argument("--gamma", type=float, default=5.0, help="Gamma value for inference")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size for the model")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training")
    parser.add_argument("--no_scheduler", action="store_true", help="Disable LR warmup and cosine schedule")
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()