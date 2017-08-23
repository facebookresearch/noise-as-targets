[[--
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.
--]]

require('nn')
require('cunn')
require('cudnn')

local arch = {}

local function makefeaturelayer(specs, dim, config)
   local features = nn.Sequential()
   -- mesure the size of the output feature
   local featsizes = torch.LongTensor({dim[2], dim[3]})
   for i = 1, #specs-1 do
      local numplanes = specs[i].numplanes
      if i == 1 then
         numplanes = 2
      end
      features:add(cudnn.SpatialConvolution(
      numplanes,
      specs[i+1].numplanes,
      specs[i].filtersize,
      specs[i].filtersize,
      specs[i].filterstride,
      specs[i].filterstride,
      specs[i].filterpadding,
      specs[i].filterpadding))

      featsizes:add(specs[i].filterpadding)
      featsizes:add(-(specs[i].filtersize - 1) / 2)
      featsizes:div(specs[i].filterstride)
      features:add(nn.SpatialBatchNormalization(specs[i + 1].numplanes))
      if specs[i].pooling ~= nil then
         features:add(
            cudnn.SpatialMaxPooling(
               specs[i].pooling.filtersize,
               specs[i].pooling.filtersize,
               specs[i].pooling.filterstride,
               specs[i].pooling.filterstride
            )
         )
         featsizes:add(-(specs[i].pooling.filtersize-1)/2)
         featsizes:div(specs[i].pooling.filterstride)
      end
      features:add(cudnn.ReLU(true))
   end
   local featsize = featsizes:prod() * specs[#specs].numplanes
   features:add(nn.View(-1, featsize))
   return features, featsize
end

local function makeclassifierlayer(specs, featsize, dropout, config)
   local classifier = nn.Sequential()
   classifier:add(nn.Linear(featsize, specs[1].numplanes))
   classifier:add(nn.BatchNormalization(specs[1].numplanes))
   classifier:add(cudnn.ReLU(true))
   if dropout and dropout > 0 then
      assert(dropout <= 1)
      classifier:add(nn.Dropout(dropout))
   end
   classifier:add(nn.Linear(specs[1].numplanes, specs[2].numplanes))
   classifier:add(nn.BatchNormalization(specs[2].numplanes))
   classifier:add(cudnn.ReLU(true))
   classifier:add(nn.Linear(specs[2].numplanes, specs[2].outputsize))
   return classifier
end

function arch.train(config)
   cudnn.fastest   = config.fastcudnn or false
   cudnn.benchmark = config.fastcudnn or false
   local dropout   = config.dropout or 0.5
   local specs = require(config.specs)(config.nclass, config.cropsize)
   local feat, featsize = makefeaturelayer(specs.feature, specs.imsize, config)
   local classifier = makeclassifierlayer(
      specs.classifier, featsize, dropout, config
   )
   local net = nn.Sequential()
   net:add(feat):add(classifier)
   net:cuda()
   return net
end

function arch.test(config, trainnet)
   cudnn.fastest   = config.fastcudnn or false
   cudnn.benchmark = config.fastcudnn or false
   if config.testsinglecrop then return trainnet end
   local testnet = nn.Sequential()
   testnet:add( nn.View(3, config.cropsize, config.cropsize) )
   testnet:add( trainnet )
   testnet:add( nn.View(10, config.nclass) )
   testnet:add( nn.Sum(2) )
   testnet:cuda() -- shared with net, because we did it already for net
   return testnet
end

function arch.parallelize(net, config)
   assert(config and config.ngpu)
   local dpt = nn.DataParallelTable(1, true, true)
   dpt:add(net:clone(), torch.range(1, config.ngpu):totable())
   return dpt
end

function arch.removeParallelTable(net)
   if torch.type(net) == 'nn.DataParallel'
   or torch.type(net) == 'nn.DataParallelTable' then
      net = net:get(1)
   end
   return net
end

function arch.criterion(config)
   return nn.CrossEntropyCriterion():cuda()
end

function arch.snapshot(net)
   -- deep clone net but 'share' weight and bias (i.e. keep a reference)
   return arch.removeParallelTable(net):clone('weight', 'bias')
end

function arch.clone(net)
   return arch.removeParallelTable(net):clone()
end

function arch.saveNamed(net, rundir, name)
   local f = torch.DiskFile(string.format('%s/%s.bin', rundir, name), 'w')
   f:binary()
   f:writeObject(arch.removeParallelTable(net):clearState())
   f:close()
end

function arch.save(net, rundir, condition)
   if not condition then return false end
   local f = torch.DiskFile(string.format('%s/model.bin', rundir), 'w')
   f:binary()
   f:writeObject(arch.removeParallelTable(net):clearState())
   f:close()
   return true
end

function arch.load(dir)
   local f = torch.DiskFile(string.format('%s/model.bin', dir))
   f:binary()
   local net = f:readObject()
   f:close()
   return arch.removeParallelTable(net)
end

function arch.loadNamed(dir, name)
   local f = torch.DiskFile(string.format('%s/%s.bin', dir, name))
   f:binary()
   local net = f:readObject()
   f:close()
   return arch.removeParallelTable(net)
end

return arch
