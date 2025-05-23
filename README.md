## Building in Visual Studio
1. Clone the repository with submodules: `git clone --recurse-submodules <repo-url>`
2. Open the `.sln` file in Visual Studio.
3. Build the solution (`Ctrl+Shift+B`).
- Eigen is included in `extern/eigen/` and configured in the project settings.

- Update submodules: git submodule update --init --recursive


#Branches:
1. Master: A feedforward neural network to classify handwritten digits (0–9) from the MNIST dataset.(Heavily commented code).
2. master-restructured-1.0: Neuron centric design, with added layer class.		
