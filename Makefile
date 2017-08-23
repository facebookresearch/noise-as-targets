# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

CC = cc
CFLAGS = -std=c99 -fPIC -Wall -Ofast -c
LDFLAGS = -shared

opt: lib

lib: hungarian.c
	$(CC) $(CFLAGS) hungarian.c
	$(CC) $(LDFLAGS) -o libhungarian.so hungarian.o

clean:
	rm -rf *.o *.so
