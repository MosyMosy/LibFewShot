# -*- coding: utf-8 -*-
import argparse
from core import Trainer
from core import Test
from core.config import Config
import os
import sys
import subprocess

sys.dont_write_bytecode = True


parser = argparse.ArgumentParser(description='LibFewShot Training')

parser.add_argument('--data_root', default='./dataset/miniImageNet--ravi',  help='path to dataset')
parser.add_argument('--shot_num', default=0, type=int, help='training num_shot')
parser.add_argument('--conf_file', type=str, required=True,  help='path to config')
parser.add_argument("--train_episode", type=int, default=300, help="train episode num")
parser.add_argument("--test_episode", type=int, default=2000, help="test episode num")
parser.add_argument("--epoch", type=int, default=100, help="test episode num")
parser.add_argument("--test_epoch", type=int, default=5, help="test episode num")
if __name__ == "__main__":
    args = parser.parse_args()

    VAR_DICT = {
        "data_root": args.data_root,
        "shot_num":args.shot_num,
        "test_shot":args.shot_num,
        "train_episode": args.train_episode,
        "test_episode": args.test_episode,
        "epoch": args.epoch,
        "test_epoch": args.test_epoch,
        "episode_size": 1,
        "test_way": 5,
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
    
    args = parser.parse_args()
    config = Config(os.path.join(result_path, "config.yaml"),
                    VAR_DICT).get_config_dict()
    test = Test(config, result_path)
    test.test_loop()
    
    
    bashCommand = "cp -r {0} ./temp".format(result_path)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()