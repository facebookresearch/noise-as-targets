#!/bin/bash
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

LUAJIT=/data/users/bojanowski/prout/bin/luajit
DATA=/data/users/bojanowski/data
MODEL=exp/pre-trained
MFILE="${MODEL}/model.bin"

if [ -f "${MFILE}" ]
then
  curl -o "${MFILE}" --create-dirs \
    https://s3-us-west-2.amazonaws.com/noise-as-targets/model.bin
done

OMP_NUM_THREADS=1 ${LUAJIT} test.lua -dataroot ${DATA} -nthread 10 \
  -maxepoch 100 -lr 0.1 -weightdecay 1e-8 -gamma 0.3 \
  -dropout 0.5 -momentum 0.9 -model ${MODEL}
