```markdown
# MyNeuralNetworkProject

* @brief A feedforward neural network.

 * Implements a multi-layer neural network for tasks like MNIST classification,
 * supporting feedforward, backpropagation, and stochastic gradient descent (SGD).
 * Supports both MSE and Cross-Entropy loss functions as well as L2 implementation.
 * Neurons: sigmod(for now).

## Prerequisites
- Visual Studio 2022 with C++ Desktop Development workload
- Python 3.x (for setup script)
- Git
- Eigen is included(as submodule) in `extern/eigen/` and configured in the project settings
- CUDA Toolkit and cuDNN (can be installed via the setup script)

## Setup Instructions
1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules <repo-url>
   ```
   Update submodules: 
   ```bash
   git submodule update --init --recursive
   ```
2. Run the setup script to configure environment variables and update PATH:
   ```bash
   python scripts/setup_dependencies.py
   ```
   *Note*: Run the script as Administrator to set environment variables and update PATH.
   
3. The script checks for CUDA and cuDNN at:
   - CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<version>`
   - cuDNN: `C:\Program Files\NVIDIA\CUDNN\<version>`
   It sets environment variables, e.g.:
   - `CUDA_PATH`: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`
   - `CUDNN_PATH`: `C:\Program Files\NVIDIA\CUDNN\v9.10`
   - `CUDNN_PATH_include`: `C:\Program Files\NVIDIA\CUDNN\v9.10\include\12.9`
   - `CUDNN_PATH_lib`: `C:\Program Files\NVIDIA\CUDNN\v9.10\lib\12.9\x64`
   - `CUDNN_PATH_bin`: `C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9`
   
4. If CUDA or cuDNN is installed in a different directory, the script will prompt for custom paths. Alternatively, manually set environment variables:
   ```bash
   setx CUDA_PATH "<your_cuda_path>"
   setx CUDNN_PATH "<your_cudnn_path>"
   setx CUDNN_PATH_include "<your_cudnn_path>\include\<version>"
   setx CUDNN_PATH_lib "<your_cudnn_path>\lib\<version>\x64"
   setx CUDNN_PATH_bin "<your_cudnn_path>\bin\<version>"
   ```
   
5. If CUDA or cuDNN is not installed, the script will provide download instructions:
   - CUDA Toolkit: [Download](https://developer.nvidia.com/cuda-downloads)
   - cuDNN: [Download](https://developer.nvidia.com/cudnn)
   Follow the script’s prompts to install and rerun.
   
6.  In Visual Studio's .vcxproj, include/library paths, post-build events,  additional dependencies and "properties->build configuration->check cuda targets" are already configured.

## Notes
- The project uses environment variables (`CUDA_PATH`, `CUDNN_PATH`, `CUDNN_PATH_include`, `CUDNN_PATH_lib`, `CUDNN_PATH_bin`) for dependency paths.
- The system PATH includes `<cuda_path>\bin` and `<cudnn_path>\bin\<version>` (set by the script).
- Eigen is included as a Git submodule in `extern/eigen/`.
- If issues persist, run the script as Administrator and follow its instructions.
- To verify the setup, build and run `src/test_cuda.cu` in Visual Studio.
