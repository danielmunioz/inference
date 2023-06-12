chmod +x *.sh

bash tools/make_fake_imagenet.sh
./download_imagenet.sh
wget https://github.com/doonny/PipeCNN/blob/392692b86747096e4324098deb38d8e93159809e/project_intel/data/imagenet/val.txt -O ILSVRC2012_img_val/val.txt

mkdir models
wget --no-check-certificate https://zenodo.org/record/2535873/files/resnet50_v1.pb -O models/resnet50_v1.pb
wget --no-check-certificate https://zenodo.org/record/2592612/files/resnet50_v1.onnx -O models/resnet50_v1.onnx
wget --no-check-certificate https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth -O models/resnet50-19c8e357.pth
wget --no-check-certificate https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5 -O models/resnet50_tf2.h5

pip install tensorflow onnxruntime cython torch torchvision tensorflow_probability
cd ../../loadgen; CFLAGS="-std=c++14" python setup.py develop; cd ../vision/classification_and_detection
python setup.py develop

git clone --recurse-submodules https://github.com/unifyai/ivy.git
pip install -e ivy