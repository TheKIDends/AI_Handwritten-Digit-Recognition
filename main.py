from runners.LeNet5Runner import LeNet5Runner
from tensorboardX import SummaryWriter
import yaml

from runners.RandomNetRunner import RandomNetRunner
from utils.utils import dict_to_namespace


def main():
    log_dir = './logs'
    logger = SummaryWriter(log_dir)

    config_path = './config/config.yml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = dict_to_namespace(config)  # Convert dictionary to Namespace object

    runner = LeNet5Runner(config, logger)
    runner.train()
    runner.test()


if __name__ == "__main__":
    main()
