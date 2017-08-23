[[--
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.
--]]

local hungarian = {}

hungarian.minCost = function(cost)
   assert(cost:dim() == 2, "Cost must be a matrix")
   assert(cost:type() == "torch.DoubleTensor", "Cost must be a double tensor")

   local ffi = require 'ffi'

   ffi.cdef[[
   int solve(const int* cost, int* res, int m, int n);
   ]]

   local libhungarian = ffi.load("libhungarian")

   local m = cost:size(1)
   local n = cost:size(2)

   local cmin = cost:min()
   local cmax = cost:max()

   local icost = cost:clone():add(-cmin):mul(1 / (cmax - cmin))
   icost = icost:mul(100000):int()

   local result = torch.IntTensor(m):zero()
   local c = libhungarian.solve(
      icost:storage():data(),
      result:storage():data(),
      m,
      n
   )
   result:add(1)
   return result
end

hungarian.maxCost = function(cost)
   return hungarian.minCost(-cost)
end

return hungarian
