# Setup

This is an instruction on how to set up the environment for GPU acceleration on a computer with NVIDIA graphics cards. 
Ubuntu is the recommended operating system. 

# Install NVIDIA driver 

List available drivers.

```sh
ubuntu-drivers devices
```

Install a driver. You should choose the one recommended by your computer (from the last command).

```sh
sudo apt install nvidia-driver-XXX
```

After having installed the driver, please reboot your computer. There might be some password setting about secure boot.

```sh
reboot
```

You can test whether the driver is successfully installed.

```sh
nvidia-smi
```

# Build virtual environment

It is recommended to run your project in an isolated virtual environment for package management.

```sh
python3 -m venv env_glia
```

Remember to activate this virtual environment every time you run your project.

```sh
source env_glia/bin/activate
```

```sh
cd GLIA
```

# Install requirements

You can install the required python packages by pip.

```sh
pip install -r requirements.txt
```
