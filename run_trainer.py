# -*- coding: utf-8 -*-
import argparse
from core import Trainer
from core import Test
from core.config import Config
import os
import sys

sys.dont_write_bytecode = True


parser = argparse.ArgumentParser(description='LibFewShot Training')

parser.add_argument('--data_root', default='./dataset/miniImageNet--ravi',  help='path to dataset')
parser.add_argument('--device_ids', default=0, help='GPU id')
parser.add_argument('--shot_num', default=0, type=int, help='training num_shot')
parser.add_argument('--conf_file', type=str,  help='path to config')
parser.add_argument("--train_episode", type=int, default=300, help="train episode num")
parser.add_argument("--test_episode", type=int, default=2000, help="test episode num")
parser.add_argument("-tag", "--tag", type=str, default='WithAffine', help="experiment tag")

if __name__ == "__main__":
    args = parser.parse_args()

    VAR_DICT = {
        "data_root": args.data_root,
        "device_ids": args.device_ids,
        "shot_num":args.shot_num,
        "test_shot":args.shot_num,
        "train_episode": args.train_episode,
        "test_episode": args.test_episode,
        "tag": args.tag,
        "epoch": 100,
        "test_epoch": 5
    }
    
    print(args.conf_file)
    config = Config(args.conf_file, variable_dict=VAR_DICT).get_config_dict()
    trainer = Trainer(config)
    result_path = trainer.train_loop()

    # Testing
    print('')
    print("----------------------------------------------------------------------------")
    print("--------------------------------   Testing   -------------------------------")
    print("----------------------------------------------------------------------------")
    VAR_DICT = {
        "test_epoch": 5,
        "device_ids": "0",
        "n_gpu": 1,
        "test_episode": 2000,
        "episode_size": 1,
        "test_way": 5,
    }
    args = parser.parse_args()
    config = Config(os.path.join(result_path, "config.yaml"),
                    VAR_DICT).get_config_dict()
    test = Test(config, result_path)
    test.test_loop()