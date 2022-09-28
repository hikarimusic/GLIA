nvidia-smi
sudo kill -9 PID

nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1