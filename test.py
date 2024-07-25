import logging
import logging.config
import os
import time

from absl import app, flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", "./configs/ddpm/mnist.py", "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", "./data", "Work directory.")
flags.DEFINE_enum("mode", "train", ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.DEFINE_string("runner", "Test", "The runner to execute")

# flags.mark_flags_as_required(["workdir", "config", "mode"])


# 'application' code
def main(argv):

    # os.chdir('E:\jikewct\Repos\ml_code')
    logging.config.fileConfig("./configs/conf/log.conf")

    logging.info("this is a log")
    logging.debug("this is a debug log")
    while True:
        logging.info("this file is running .......")
        logging.info(FLAGS.mode)
        time.sleep(10)


"""
    if FLAGS.mode == "train":
        print(FLAGS.config)
    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        print(FLAGS.config)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")
"""

if __name__ == "__main__":
    app.run(main)
