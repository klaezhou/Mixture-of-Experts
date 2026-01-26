import torch
import torch.optim as optim

class Trainer:
    """
    Trainer Module:
    ---------------
    1) Orchestrates the training loop using the injected model and equation.
    2) Polymorphism: calls eq.compute_loss(model, batch) regardless of PDE type.
    """
    def __init__(self, model, equation, args):
        self.model = model
        self.eq = equation
        self.args = args

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # scheduler is optional
        self.use_scheduler = getattr(self.args, "use_scheduler", True)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=getattr(self.args, "lr_step_size", 1000),
            gamma=getattr(self.args, "lr_gamma", 0.5)
        )

    def train_step(self, data):
        """
        One training step using a FIXED batch 'data'.
        data: dict returned by eq.get_data(...)
        """
        self.optimizer.zero_grad(set_to_none=True)

        loss_out = self.eq.compute_loss(self.model, data)

        # expect dict with "total", but keep it robust
        if isinstance(loss_out, dict):
            total_loss = loss_out["total"]
            loss_dict = loss_out
        else:
            total_loss = loss_out
            loss_dict = {"total": total_loss}

        total_loss.backward()
        self.optimizer.step()

        if self.use_scheduler and self.scheduler is not None:
            self.scheduler.step()

        # convert for logging (Tensor -> float)
        log_losses = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                log_losses[k] = float(v.detach().cpu())
            else:
                log_losses[k] = float(v)

        return log_losses

    def train(self, data, epochs):
        self.model.train()

        log_freq = getattr(self.args, "log_freq", 200)
        for epoch in range(epochs):
            losses = self.train_step(data)

            # epoch is 0-based; print at 1-based step
            it = epoch + 1
            if it % log_freq == 0:
                self._print_log(it, losses)

    def _print_log(self, it, losses):
        """print training log"""
        total = losses.get("total", None)
        if total is None:
            raise KeyError("loss dict must contain key 'total'")

        log_str = f"[Iter {it:05d}] Total: {total:.6e}"

        for k in sorted(losses.keys()):
            if k == "total":
                continue
            log_str += f" | {k.upper()}: {losses[k]:.6e}"

        lr = self.optimizer.param_groups[0]["lr"]
        log_str += f" | LR: {lr:.3e}"

        print(log_str)
