# -*- coding: utf-8 -*-
from core import Test
from core.config import Config
import os
import sys
import argparse
sys.dont_write_bytecode = True


parser = argparse.ArgumentParser(description='LibFewShot Testing')
parser.add_argument("-result_method", "--result_method",
                    default='', help="a method result path")
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "2",
    "n_gpu": 1,
    "test_episode": 2000,
    "episode_size": 1,
    "test_way": 5,
}

if __name__ == "__main__":
    args = parser.parse_args()
    config = Config(os.path.join("./results/" + args.result_method, "config.yaml"),
                    VAR_DICT).get_config_dict()
    test = Test(config, "./results/" + args.result_method)
    test.test_loop()
