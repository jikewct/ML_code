import argparse
import logging
import logging.config
import os
import sys
import time
import traceback
from sys import argv

from absl import app, flags
from ml_collections.config_flags import config_flags

# python.exe .\main.py --runner TrainDiffusion --mode test  --config .\configs\ddim\cifar10.py

## 通过torch.cuda.get_device_capability()获取
cuda_arch_list = "6.1+PTX"
os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch_list
os.system(f'set TORCH_CUDA_ARCH_LIST={cuda_arch_list}')
#os.chdir('E:\jikewct\Repos\ml_code')

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "./configs/ddpm/mnist.py", "Training configuration.", lock_config=False)
flags.DEFINE_enum("mode", "train", ["train", "eval", "test", "debug_sampling"], "Running mode: train, eval or test")
flags.DEFINE_string("runner", "Test", "The runner to execute")


def main(argv):
    logging.config.fileConfig("./configs/conf/log.conf")
    config = FLAGS.config
    logging.debug(config)
    try:
        from scripts.train_diffusion import TrainDiffusion
        runner = eval(FLAGS.runner)(FLAGS, config)
        runner.run()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    app.run(main)
