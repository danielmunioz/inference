./run_local.sh pytorch-native resnet50 cpu --dataset imagenet_pytorch_native --profile resnet50-pytorch-native --accuracy --data-format NCHW --qps 20 --count 64 --time 30
# TF2
# ./run_local.sh tf2 resnet50 cpu --dataset imagenet_tf2 --accuracy

# torch2jax
# ./run_local.sh torch2jax resnet50 cpu --dataset imagenet_torch2jax --profile resnet50-torch2jax --accuracy