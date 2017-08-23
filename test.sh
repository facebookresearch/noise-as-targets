#!/bin/bash
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

LUAJIT=/data/users/bojanowski/prout/bin/luajit

MODEL=exp/unsup-dim-2048-perm-3-lr-1.0e-02/epoch-00010
DATA=/data/users/bojanowski/data

OMP_NUM_THREADS=1 ${LUAJIT} test.lua -dataroot ${DATA} -nthread 10 \
  -maxepoch 100 -lr 0.1 -weightdecay 1e-8 -gamma 0.3 \
  -dropout 0.5 -momentum 0.9 -model ${MODEL}
