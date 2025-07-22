
import os
import subprocess
import winreg
import glob
import sys
import re

def check_winget():
    try:
        subprocess.run(["winget", "--version"], capture_output=True, check=True)
        return True
    except FileNotFoundError:
        print("winget not found large language model. Please ensure Windows Package Manager is installed.")
        return False

def prompt_for_path(component, default_path):
    print(f"\n{component} not found at {default_path}.")
    print(f"Please provide the {component} installation path or press Enter to use default ({default_path}):")
    user_input = input().strip()
    return user_input if user_input else default_path

def find_latest_version(base_dir):
    if not os.path.exists(base_dir):
        return None
    versions = glob.glob(os.path.join(base_dir, "v*"))
    if not versions:
        return None
    return max(versions, key=os.path.getmtime)

def find_cudnn_library_version(cudnn_base, subdir):
    subdir_path = os.path.join(cudnn_base, subdir)
    if not os.path.exists(subdir_path):
        return None
    versioned_dirs = glob.glob(os.path.join(subdir_path, "*"))
    if not versioned_dirs:
        return None
    return max(versioned_dirs, key=os.path.getmtime)

def set_environment_variable(name, value):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_ALL_ACCESS) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_EXPAND_SZ, value)
        print(f"Set {name} to {value}")
        return True
    except PermissionError:
        print(f"Permission denied. Please run as Administrator to set {name}.")
        print(f"Manually set: setx {name} \"{value}\"")
        return False
    except Exception as e:
        print(f"Error setting {name}: {e}")
        print(f"Manually set: setx {name} \"{value}\"")
        return False

def update_path_env(cuda_bin, cudnn_bin):
    current_path = os.environ.get("PATH", "")
    paths_to_add = [cuda_bin, cudnn_bin]
    new_paths = [p for p in paths_to_add if p and p not in current_path]
    if not new_paths:
        print("All required paths already in PATH.")
        return True
    
    print("Updating PATH environment variable...")
    new_path = f"{current_path};{' '.join(new_paths)}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", 0, winreg.KEY_ALL_ACCESS) as key:
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
        print("PATH updated successfully. Please restart your terminal or system to apply changes.")
        return True
    except PermissionError:
        print("Permission denied. Please run as Administrator to update PATH.")
        print(f"Manually add to PATH: {' '.join(new_paths)}")
        return False
    except Exception as e:
        print(f"Error updating PATH: {e}")
        print(f"Manually add to PATH: {' '.join(new_paths)}")
        return False

def update_submodules():
    print("Updating Git submodules (e.g., Eigen)...")
    try:
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
        print("Submodules updated successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to update submodules: {e}")
        return False

def main():
    print("Setting up dependencies for MyNeuralNetworkProject...")
    success = True

    # Step 1: Check Git submodules
    if not update_submodules():
        success = False

    # Step 2: Check CUDA
    cuda_default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(os.path.join(cuda_path, "bin")):
        print(f"CUDA found at {cuda_path}")
    else:
        cuda_path = find_latest_version(cuda_default)
        if not cuda_path:
            cuda_path = prompt_for_path("CUDA", cuda_default)
            if not os.path.exists(os.path.join(cuda_path, "bin")):
                print("CUDA not found. Please download and install from https://developer.nvidia.com/cuda-downloads")
                if check_winget():
                    print("Alternatively, run: winget install --id Nvidia.CUDA --accept-source-agreements --accept-package-agreements")
                success = False
                return
        set_environment_variable("CUDA_PATH", cuda_path)

    # Extract CUDA version (e.g., v12.9)
    cuda_version = os.path.basename(cuda_path)
    cuda_bin = os.path.join(cuda_path, "bin")
    print(f"CUDA version: {cuda_version}")
    print(f"CUDA include: {os.path.join(cuda_path, 'include')}")
    print(f"CUDA lib: {os.path.join(cuda_path, 'lib', 'x64')}")
    print(f"CUDA bin: {cuda_bin}")

    # Step 3: Check cuDNN
    cudnn_default = r"C:\Program Files\NVIDIA\CUDNN"
    cudnn_path = find_latest_version(cudnn_default)
    if not cudnn_path:
        cudnn_path = prompt_for_path("cuDNN", cudnn_default)
        if not os.path.exists(cudnn_path):
            print("cuDNN not found. Please download and install from https://developer.nvidia.com/cudnn")
            success = False
            return
    cudnn_version = os.path.basename(cudnn_path)
    set_environment_variable("CUDNN_PATH", cudnn_path)
    print(f"cuDNN version: {cudnn_version}")

    # Step 4: Resolve cuDNN library versions
    cudnn_include = find_cudnn_library_version(cudnn_path, "include")
    cudnn_lib = find_cudnn_library_version(cudnn_path, "lib")
    cudnn_bin = find_cudnn_library_version(cudnn_path, "bin")
    if not (cudnn_include and cudnn_lib and cudnn_bin):
        print("Failed to find cuDNN library versioned subdirectories (include, lib, bin).")
        success = False
        return

    cudnn_lib = os.path.join(cudnn_lib, "x64")  # Append x64 for lib
    if not os.path.exists(os.path.join(cudnn_lib, "cudnn.lib")):
        print(f"cuDNN library not found at {cudnn_lib}")
        success = False
        return

    # Set cuDNN version-specific environment variables
    library_version = os.path.basename(cudnn_include)
    set_environment_variable("CUDNN_PATH_include", cudnn_include)
    set_environment_variable("CUDNN_PATH_lib", cudnn_lib)
    set_environment_variable("CUDNN_PATH_bin", cudnn_bin)
    print(f"cuDNN library version: {library_version}")
    print(f"cuDNN include: {cudnn_include}")
    print(f"cuDNN lib: {cudnn_lib}")
    print(f"cuDNN bin: {cudnn_bin}")

    # Step 5: Update PATH
    if not update_path_env(cuda_bin, cudnn_bin):
        success = False

    # Step 6: Final output
    if success:
        print("\nDependencies configured successfully!")
        print(f"CUDA path: {cuda_path} (Env: CUDA_PATH)")
        print(f"cuDNN path: {cudnn_path} (Env: CUDNN_PATH)")
        print(f"cuDNN include: {cudnn_include} (Env: CUDNN_PATH_include)")
        print(f"cuDNN lib: {cudnn_lib} (Env: CUDNN_PATH_lib)")
        print(f"cuDNN bin: {cudnn_bin} (Env: CUDNN_PATH_bin)")
        print("Open MyNeuralNetwork.vcxproj in Visual Studio, configure include/library paths, and build.")
    else:
        print("\nSetup incomplete. Please follow the instructions above to resolve issues.")

if __name__ == "__main__":
    main()
