import logging

from torch.utils.tensorboard import SummaryWriter

from kanachan.training.common import get_distributed_environment

logger = logging.getLogger(__name__)


class TrainingProgress:
    """Report completed optimization rounds separately from consumed samples."""

    def __init__(
        self,
        writer: SummaryWriter,
        batch_size: int,
        accumulation_steps: int,
        target_interval: int | None = None,
        target_rate: float | None = None,
        *,
        initial_optimizer_steps: int = 0,
    ) -> None:
        world_size, rank, _ = get_distributed_environment()
        self.optimizer_steps = 0
        self._initial_optimizer_steps = initial_optimizer_steps
        self.target_updates = 0
        self._writer = writer
        self._primary = rank == 0
        if self._primary:
            logger.info(
                "Optimizer update every %d batches (%d samples globally).",
                accumulation_steps,
                batch_size * world_size * accumulation_steps,
            )
            if target_interval is not None:
                logger.info(
                    "Target update every %d completed optimizer steps; "
                    "source weight per update = %s.",
                    target_interval,
                    target_rate,
                )

    @property
    def total_optimizer_steps(self) -> int:
        return self._initial_optimizer_steps + self.optimizer_steps

    def optimizer_step(self, num_samples: int) -> None:
        self.optimizer_steps += 1
        if self._primary:
            self._writer.add_scalar(
                "Optimizer steps (this run)", self.optimizer_steps, num_samples
            )

    def target_update(self, num_samples: int) -> None:
        self.target_updates += 1
        if self._primary:
            self._writer.add_scalar(
                "Target updates (this run)", self.target_updates, num_samples
            )

    def finish(self, pending_batches: int) -> None:
        if self.optimizer_steps == 0:
            raise RuntimeError(
                "No optimizer updates completed. Check for empty or exhausted "
                "data and non-finite gradients. To reread a completed epoch, "
                "resume with start_new_epoch=true."
            )
        if not self._primary:
            return
        logger.info(
            "Completed in this run: %d optimizer steps, %d target updates.",
            self.optimizer_steps,
            self.target_updates,
        )
        if pending_batches:
            logger.warning(
                "%d trailing batches did not complete a gradient "
                "accumulation window and were not applied.",
                pending_batches,
            )
