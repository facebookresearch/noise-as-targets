# Unsupervised Learning by Predicting Noise

This is the code used for unsupervised training of convolutional neural networks as described in the ICML 2017 paper Unsupervised Learning by Predicting Noise ([arXiv](https://arxiv.org/abs/1704.05310)).

The code is composed of two modules, one for unsupervised feature learning, and one for training a supervised classifier on top of them.

## Requirements

The code is in Lua Torch, and therefore requires a working torch installation.
It was tested with a LuaJIT installation obtained using:
```
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/your/prefix
```
where `/your/prefix` is the path where you want a local install of LuaJIT and all dependencies.
In order to run this code, you will need the following modules:
```
torch
cutorch
nn
cunn
cudnn
torchnet
image
```
You can install them using:
```
$ /your/prefix/bin/luarocks install [package]
```

## Compiling the C code

Computing the optimal permutation of targets requires solving an assignment problem.
We do so using the Hungarian algorithm, also known as the Kuhn-Munkres algorithm.
We provide a C implementation and the corresponding Lua interface (using ffi).
In order to use it, please run `$ make` which will compile the C code and create the shared library.

## Getting data

This code uses an `IndexedDataset` to load data.
This is a TorchNet data structure which provides an efficient way to bundle data into a single archive file, associated with an indexed file.
Given a copy of ImageNet, you can create the archive using:
```
$ ./make-index.sh
```
Inside this script, please replace the source and destination paths as follows:
```
IMAGENET=/data/users/bojanowski/data/imagenet-full-size
DEST=/data/users/bojanowski/data/imagenet-idx
```
Where `/data/users/bojanowski/data/imagenet-full-size` is where you store your copy of ImageNet images:
```
$ ls -1 /data/users/bojanowski/data/imagenet-full-size
labels.txt
synset_words.txt
test
train
val
```
This will produce indexed archives (roughly 70Gb for training images) in the `${DEST}` folder.

## Running the unsupervised training

Unsupervised training can be launched by running:
```
$ ./main.sh
```
Please specify the location of your LuaJIT installation in the script:
```
LUAJIT=/data/users/bojanowski/local/bin/luajit
```
Please also provide the path to the data folder where the `imagenet-idx` folder is located:
```
DATA=/data/users/bojanowski/data
```
Finally, you can specify where you want to save the logs, checkpoints and models using:
```
EXP=exp
```
During training, checkpoints and logs will be saved to the directory specified in `${EXP}`.
A new directory is created for each new set of parameters.
This directory of results contains the following things:
```
$ ls -1 exp/unsup-dim-2048-perm-3-lr-1.0e-02
checkpoint.bin
checkpointModel.bin
config.bin
log-20170727-022756.txt
rcp-0-checkpoint.bin
rcp-0-checkpointModel.bin
rcp-1-checkpoint.bin
rcp-1-checkpointModel.bin
rcp-2-checkpoint.bin
rcp-2-checkpointModel.bin
epoch-00010
epoch-00020
```
We have implemented rolling checkpoints, where a version of the model and the corresponding codes are saved every epoch, in a rolling fashion.
The files `checkpoint.bin` and `checkpointModel.bin` are symbolic links to the latest checkpoint available.
Models are also saved every other k epochs (set using the `-saveperiod` flag), and can be found in for instance in `epoch-00010'.

A complete list of options can be obtained using:
```
$ luajit main.lua -h
```

## Learning the MLP on the supervised task

Once the unsupervised training is finished, you can launch the transfer task using:
```
$ ./test.sh
```
You need to specify the path to the ImageNet data and the directory in which your model lies.
The code should run the supervised training of the MLP in the AlexNet and log the train and val performance along epochs.
The output of this code will be saved in a subdirectory of where the model is.

The full set of parameters for this supervised training can be obtained using:
```
$ luajit test.lua -h
```

## ImageNet classification with a pre-trained model

We provide a pre-trained model, available for download [here](https://dl.fbaipublicfiles.com/noise-as-targets/model.bin).
You can run the transfer task with a pre-trained model by running:
```
$ ./test-pretrained.sh
```
This will download the pre-trained model and launch the transfer learning on ImageNet.
