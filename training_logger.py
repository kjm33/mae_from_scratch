import time
from contextlib import contextmanager

import torch
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn, SpinnerColumn


class TrainingLogger:
    """Handles all progress display, stats tracking, and profiling for a training run.

    Usage:
        with TrainingLogger(device, num_epochs, steps_per_epoch, profile_tag) as logger:
            for epoch in range(num_epochs):
                logger.begin_epoch(epoch)
                for step, batch in enumerate(dataloader):
                    # ... forward / backward / optimizer ...
                    logger.on_step(loss.item())
                logger.end_epoch()

            # After training, profile a single step with NVTX section annotations:
            with logger.profile_step():
                with logger.section("forward"):
                    loss, _, _ = model(batch)
                with logger.section("backward"):
                    loss.backward()
                with logger.section("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()
    """

    def __init__(self, device, num_epochs, steps_per_epoch, profile_tag):
        self.device = device
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.profile_tag = profile_tag

        self.console = Console()
        self._progress = None
        self._task = None
        self._prof = None

        # cumulative stats
        self.train_start = None
        self.max_vram_mb = 0.0
        self.total_loss = 0.0
        self.total_steps = 0

        # per-epoch state
        self._epoch_loss = 0.0
        self._epoch_steps = 0
        self._epoch_start = None

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.train_start = time.time()
        self._start_profiler()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Epoch {task.fields[epoch]}/{task.fields[num_epochs]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss [green]{task.fields[loss]:.4f}"),
            TextColumn("VRAM [yellow]{task.fields[vram]:.1f} GB"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._progress.__enter__()
        self._task = self._progress.add_task(
            "train", total=self.steps_per_epoch,
            epoch=1, num_epochs=self.num_epochs, loss=0.0, vram=0.0,
        )
        return self

    def __exit__(self, *_):
        self._progress.__exit__(None, None, None)
        if self._prof is not None:
            self._prof.stop()
        self._print_summary()

    # ------------------------------------------------------------------
    # per-epoch hooks
    # ------------------------------------------------------------------

    def begin_epoch(self, epoch):
        self._epoch_loss = 0.0
        self._epoch_steps = 0
        self._epoch_start = time.time()
        self._progress.reset(self._task, total=self.steps_per_epoch)
        self._progress.update(self._task, epoch=epoch + 1)

    def end_epoch(self, epoch):
        avg_loss = self._epoch_loss / self._epoch_steps if self._epoch_steps else 0.0
        epoch_time = time.time() - self._epoch_start
        self.console.print(
            f"[bold]Epoch {epoch + 1}/{self.num_epochs}[/bold] "
            f"avg_loss=[green]{avg_loss:.4f}[/green] "
            f"time=[cyan]{epoch_time:.1f}s[/cyan]"
        )

    # ------------------------------------------------------------------
    # per-step hook
    # ------------------------------------------------------------------

    def on_step(self, loss_val):
        self._epoch_loss += loss_val
        self._epoch_steps += 1
        self.total_loss += loss_val
        self.total_steps += 1

        vram_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        self.max_vram_mb = max(self.max_vram_mb, vram_mb)
        self._progress.update(self._task, advance=1, loss=loss_val, vram=vram_mb / 1024)

        self._step_profiler()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _start_profiler(self):
        self._prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=tensorboard_trace_handler(f"./runs/{self.profile_tag}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            acc_events=True,
        )
        self._prof.start()

    def _step_profiler(self):
        if self._prof is None:
            return
        self._prof.step()
        if self.total_steps >= 5:  # wait(1) + warmup(1) + active(3)
            self._prof.stop()
            self._prof = None

    # ------------------------------------------------------------------
    # single-step profiling with NVTX section annotations
    # ------------------------------------------------------------------

    @contextmanager
    def profile_step(self):
        """Profile a single training step. Wrap the step body in this context manager.

        The profiler records CPU + CUDA activity, shapes, memory, and call stacks,
        and writes the trace to ``./runs/<profile_tag>/``.
        Use ``section()`` inside to annotate forward / backward / optimizer phases.
        """
        self.console.print("[bold cyan]Profiling single step...[/bold cyan]")
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler(f"./runs/{self.profile_tag}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            acc_events=True,
        )
        with prof:
            with torch.profiler.record_function("train_step"):
                yield
        self.console.print(
            f"[bold green]Trace saved to ./runs/{self.profile_tag}/[/bold green]"
        )

    @contextmanager
    def section(self, name: str):
        """Mark an NVTX range visible in the profiler trace and TensorBoard."""
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def _print_summary(self):
        elapsed = time.time() - self.train_start
        table = Table(title=f"Training Summary — {self.profile_tag}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_row("Total time", f"{elapsed / 3600:.2f} h  ({elapsed:.0f} s)")
        table.add_row("Total steps", str(self.total_steps))
        table.add_row("Avg steps/sec", f"{self.total_steps / elapsed:.2f}")
        table.add_row("Peak VRAM", f"{self.max_vram_mb / 1024:.2f} GB  ({self.max_vram_mb:.0f} MB)")
        table.add_row("Avg loss", f"{self.total_loss / self.total_steps:.4f}" if self.total_steps else "n/a")
        self.console.print(table)
