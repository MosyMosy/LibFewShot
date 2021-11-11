# -*- coding: utf-8 -*-
import sys
import argparse
sys.dont_write_bytecode = True

import os

from core.config import Config
from core import Trainer


parser = argparse.ArgumentParser(description='LibFewShot Testing')
parser.add_argument("-resume_path", "--resume_path",
                    default='', help="a method result path")


if __name__ == "__main__":
    args = parser.parse_args()
    PATH = args.resume_path
    config = Config(os.path.join(PATH, "config.yaml"), is_resume=True).get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
