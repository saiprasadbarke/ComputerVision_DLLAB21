from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Evaluation:
    def __init__(self, store_dir, name, stats=[]):
        """
        Creates placeholders for the statistics listed in stats to generate tensorboard summaries.
        e.g. stats = ["loss"]
        """
        log_dir = f"{store_dir}/{name}"
        self.tf_writer = SummaryWriter(
            log_dir=log_dir,
            filename_suffix=f"_{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        print(f"tensorboard --logdir={log_dir}")

    def write_episode_data(self, epoch, eval_dict):
        """
        Write episode statistics in eval_dict to tensorboard, make sure that the entries in eval_dict are specified in stats.
        e.g. eval_dict = {"loss" : 1e-4}
        """
        for metric, value in eval_dict.items():

            self.tf_writer.add_scalar(metric, value, epoch)
            self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
