

import torch
import torch.backends
import torch.backends.cudnn
import torch.utils
import torch.utils.cpp_extension
import torch.version
from absl import app


def main(argv):

    print(
        f"torch:{torch.__version__}, cuda:{torch.version.cuda}, cudnn:{torch.backends.cudnn.version()}, cuda_home:{torch.utils.cpp_extension.CUDA_HOME}, cudnn_home:{torch.utils.cpp_extension.CUDNN_HOME}, rocm_home:{torch.utils.cpp_extension.ROCM_HOME}"
    )


if __name__ == "__main__":
    app.run(main)
