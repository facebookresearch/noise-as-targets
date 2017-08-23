[[--
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.
--]]

local argcheck = require 'argcheck'

return argcheck{
   {name = 'nclass', type = 'number'},
   {name = 'dim',    type = 'number'},
   call = function (nclass, dim)
      return {
         imsize = {3, dim, dim},
         feature = {
            {
               filtersize=11, filterstride=4, filterpadding=2, numplanes=3,
               pooling = {filtersize=3, filterstride=2}
            },
            {
               filtersize=5, filterstride=1, filterpadding=2, numplanes=96,
               pooling = {filtersize=3, filterstride=2}
            },
            {
               filtersize=3, filterstride=1, filterpadding=1, numplanes=256
            },
            {
               filtersize=3, filterstride=1, filterpadding=1, numplanes=384
            },
            {
               filtersize=3, filterstride=1, filterpadding=1, numplanes=384,
               pooling = {filtersize=3, filterstride=2}
            },
            {
               numplanes=256
            },
         },
            classifier = {
                {numplanes=4096},
                {numplanes=4096, outputsize=nclass}
            }
        }
    end
}
