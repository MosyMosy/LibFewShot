# -*- coding: utf-8 -*-
from core import Trainer
from core import Test
from core.config import Config
import os
import sys
import argparse
sys.dont_write_bytecode = True


parser = argparse.ArgumentParser(description='LibFewShot Testing')
parser.add_argument("-resume_path", "--resume_path",
                    default='', help="a method result path")


if __name__ == "__main__":
    args = parser.parse_args()
    config = Config(os.path.join(args.resume_path, "config.yaml"),
                    is_resume=True).get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()

    # Testing
    print('')
    print("----------------------------------------------------------------------------")
    print("--------------------------------   Testing   -------------------------------")
    print("----------------------------------------------------------------------------")

    VAR_DICT = {
        "test_epoch": 5,
        "n_gpu": 1,
        "test_episode": 2000,
        "episode_size": 1,
        "test_way": 5,
    }
    args = parser.parse_args()
    config = Config(os.path.join(args.resume_path, "config.yaml"),
                    VAR_DICT).get_config_dict()
    test = Test(config, args.resume_path)
    test.test_loop()