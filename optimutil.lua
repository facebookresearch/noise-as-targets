[[--
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.
--]]

local optimutil = {}

optimutil.weightDecay = function(network, skip)
   skip = skip or {'nn.BatchNormalization','nn.SpatialBatchNormalization'}
   print(network)
   local modules = network:listModules()
   for i, m in pairs(modules) do
      for _, t in pairs(skip) do
         if torch.isTypeOf(m, t) then modules[i] = nil end
      end
   end
   return function(decay)
      assert(decay >= 0, 'decay should be positive!')
      if decay == 0 then return end
      for _, m in pairs(modules) do
         if m.weight then m.weight:mul(1 - decay) end
      end
   end
end

optimutil.momentum = function(network)
   local dm = {}
   local _, dw = network:parameters()
   for i, g in ipairs(dw) do
      dm[i] = g.new(g:size()):zero()
   end
   return function(mom)
      assert(mom and type(mom) == 'number' and mom >= 0)
      if mom == 0 then return dw end
      for i, g in ipairs(dw) do
         dm[i]:mul(mom):add(1 - mom, g)
         g:copy(dm[i])
      end
   end
end

return optimutil
