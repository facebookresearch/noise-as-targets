#!/bin/bash
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

LUAJIT=/data/users/bojanowski/prout/bin/luajit

mkdir -p exp

EXP=exp
DATA=/data/users/bojanowski/data

OMP_NUM_THREADS=1 ${LUAJIT} main.lua -rundir ${EXP} -ngpu 2 -gpu 1 \
  -dataroot ${DATA} -nthread 24 -maxepoch 100 -permute 3 -dim 2048
