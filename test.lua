[[--
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.
--]]

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
local tnt = require 'torchnet'
local paths = require 'paths'

cudnn.fastest   = true
cudnn.benchmark = true

-- reading the configuration file
local cmd = torch.CmdLine()
cmd:option('-dataroot', '', 'path to folder containing imagenet-idx')
cmd:option('-model', '', 'where to load the unsup model from')
cmd:option('-nthread', 20, 'number of dataset iterator threads')
cmd:option('-gpu', 1, 'id of GPU to use')
cmd:option('-seed', 1111, 'random seed')
cmd:option('-batch', 256, 'batch size')
cmd:option('-testbatch', 64, 'batch size (test)')
cmd:option('-maxepoch', 1, 'number of training epochs')
cmd:option('-maxload', -1, 'number of training examples')
cmd:option('-testsinglecrop', 0, 'whether or not use single crop at test time')
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-gamma', 0.3, 'dacay of learning rate')
cmd:option('-dropout', 0.5, 'dropout rate')
cmd:option('-weightdecay', 1e-8, 'weight decay')
cmd:option('-momentum', 0.9, 'momentum')
local config = cmd:parse(arg)
assert(config.model ~= '', 'Please provide path to model')
assert(config.dataroot ~= '', 'Please provide path to data')

-- various inits
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)

config.datadir = string.format('%s/imagenet-idx', config.dataroot)
config.loader = 'imagenet'
config.nclass = 1000
config.imsize  = 256
config.cropsize = 224
config.arch = 'alexnet'
config.specs = 'alexnet-specs'
config.dataidx = true

if config.maxload == -1 then config.maxload = nil end
if config.testsinglecrop == 0 then
   config.testsinglecrop = false
else
   config.testsinglecrop = true
end

-- setup running directory
local rundir = string.format(
   '%s/test-lr-%3.1e-g-%3.1e-wd-%3.1e-do-%3.1f-m-%3.1f-sc-%d',
   config.model,
   config.lr,
   config.gamma,
   config.weightdecay,
   config.dropout,
   config.momentum,
   config.testsinglecrop and 1 or 0
)
tnt.utils.sys.mkdir(rundir)
torch.save(string.format('%s/config.bin', rundir), config)

-- get architecture
local arch = require(config.arch)

-- load features from unsup model
local mainnet = arch.loadNamed(config.model, 'model')
local feature = nn.Sequential()
feature:add(mainnet:get(1))
feature:add(mainnet:get(2))
feature:cuda()
feature:evaluate()

-- if checkpoint is available, load it - otherwise get fresh MLP
local checkpointFname = string.format('%s/checkpoint.bin', rundir)
local classifier, checkpoint
if paths.filep(checkpointFname) then
   io.write(string.format("| Found checkpoint at %s.", checkpointFname))
   checkpoint = torch.load(checkpointFname)
   classifier = arch.loadNamed(rundir, 'checkpointModel'):cuda()
else
   classifier = arch.train(config)
   classifier = classifier:get(2)
   classifier:cuda()
end

-- build the test network (with multiple crops)
local testnet
local tempnet = nn.Sequential()
tempnet:add(feature)
tempnet:add(classifier)
if config.testsinglecrop then
   testnet = tempnet
else
   testnet = arch.test(config, tempnet)
end

-- setup criterion
local crit = nn.CrossEntropyCriterion():cuda()

-- training dataset
local train = tnt.ParallelDatasetIterator{
   nthread = config.nthread,
   init = function()
      require 'cutorch'
   end,
   closure = function()
      local tnt       = require 'torchnet'
      return tnt.BatchDataset{
         dataset = tnt.ShuffleDataset{
            dataset = require(config.loader)(config, 'train'),
            replacement = false,
         },
         batchsize = config.batch,
         policy = 'skip-last',
      }
   end,
   ordered = true
}

-- test dataset
local test = tnt.ParallelDatasetIterator{
   nthread = config.nthread,
   init = function()
      require 'cutorch'
   end,
   closure = function()
      local tnt = require 'torchnet'
      local mode = config.dataset == 'dirty' and 'test' or 'valid'
      return tnt.BatchDataset{
         dataset = tnt.ShuffleDataset{
            dataset = require(config.loader)(config, mode),
            replacement = false,
         },
         batchsize = config.testbatch,
         policy = 'skip-last',
      }
   end,
   ordered = true
}

-- setup meters
local timer = tnt.TimeMeter{unit = true}
local loss = tnt.AverageValueMeter()
local trainerr = tnt.ClassErrorMeter{topk = {5,1}}
local testerr = tnt.ClassErrorMeter{topk = {5,1}}
local trainsetsize = train:execSingle('size')
local testsetsize = test:execSingle('size')
local minerr = math.huge
local minlr = config.minlr or 1e-5

-- function to send data to the gpu:
local memongpu = {}
local send2gpu = function(sample)
   for k, v in pairs(sample) do
      if not memongpu[k] then
         memongpu[k] = torch.CudaTensor()
      end
      memongpu[k]:resize(v:size()):copy(v)
      sample[k] = memongpu[k]
   end
end

-- get WD + momentum function
local engine = tnt.SGDEngine()
local optimutil = require 'optimutil'
print(classifier)
local applyMomentum = optimutil.momentum(classifier)
local applyWeightDecay = optimutil.weightDecay(classifier)

-- setup logger
local logtext = require 'torchnet.log.view.json'
local logkeys = {
   "model",
   "epoch",
   "ilr",
   "lr",
   "gamma",
   "weightDecay",
   "dropout",
   "time",
   "loss",
   "trainAcc",
   "trainErr",
   "testAcc",
   "testErr",
   "testsinglecrop"
}
local logformat = {
   "%s",
   "%d",
   "%4.2e",
   "%4.2e",
   "%4.2e",
   "%4.2e",
   "%4.2f",
   "%4.2f",
   "%6.4f",
   "%4.2f",
   "%4.2f",
   "%4.2f",
   "%4.2f",
   "%d"
}
local runhash = os.date("%Y%m%d-%H%M%S")
local log = tnt.Log{
   keys = logkeys,
   onFlush = {
      logtext{
         filename = string.format('%s/log-%s.txt', rundir, runhash),
         keys = logkeys,
         format = logformat
      }
   }
}

-- zero meters and take care of LR decay
function engine.hooks.onStartEpoch(state)
   loss:reset()
   trainerr:reset()
   timer:reset()
   timer:resume()
   state.t = 0
   if checkpoint and state.epoch < checkpoint.epoch then
      state.epoch = checkpoint.epoch
   end
   if state.epoch >= 20 then
      state.lr = config.lr / (1 + config.gamma * (state.epoch - 20))
   end
   collectgarbage()
   collectgarbage()
end

-- send to GPU and forward through features
function engine.hooks.onSample(state)
   if type(state.sample.target) == 'table' then
      state.sample.target = torch.Tensor(state.sample.target)
   end
   send2gpu(state.sample)
   state.sample.input = feature:forward(state.sample.input)
end

-- account for loss
function engine.hooks.onForwardCriterion(state)
   loss:add(state.criterion.output)
   trainerr:add(state.network.output, state.sample.target)
   timer:incUnit()
end

-- WD + momentum
function engine.hooks.onBackward(state)
   applyMomentum(config.momentum)
   applyWeightDecay(config.weightdecay)
end

-- print log
function engine.hooks.onUpdate(state)
   if state.t % 10 == 0 then
      io.write(
         string.format(
            '\tepoch=%-5d batch=%-5d/%-5d loss=%-10.5f ms/b=%-7.0f\n',
            state.epoch,
            state.t,
            trainsetsize,
            loss:value(),
            1000 * timer:value()
         )
      )
   end
   collectgarbage()
   collectgarbage()
end

-- test loop
local function testeval(network, iterator, testerr)
   local testengine = tnt.SGDEngine()

   function testengine.hooks.onStart(state)
      testerr:reset()
      state.t = 0
   end

   function testengine.hooks.onSample(state)
      if type(state.sample.target) == 'table' then
         state.sample.target = torch.Tensor(state.sample.target)
      end
      send2gpu(state.sample)
   end

   function testengine.hooks.onForward(state)
      testerr:add(state.network.output, state.sample.target)
      collectgarbage()
   end

   testengine:test{
      network  = network,
      iterator = iterator
   }
end

-- carry out test and print logs
function engine.hooks.onEndEpoch(state)
   timer:stop()
   testeval(testnet, test, testerr)

   arch.saveNamed(state.network, rundir, 'checkpointModel')
   local toCheckpoint = {}
   toCheckpoint.epoch = state.epoch
   toCheckpoint.seed = config.seed
   torch.save(checkpointFname, toCheckpoint)

   -- spit out log
   log:set{
      model = config.model,
      epoch = state.epoch,
      ilr = config.lr,
      lr = state.lr,
      gamma = config.gamma,
      weightDecay = config.weightdecay,
      dropout = config.dropout,
      time = timer:value() * 1000,
      loss = loss:value(),
      trainAcc = 100 - trainerr:value(1),
      trainErr = trainerr:value(1),
      testAcc = 100 - testerr:value(1),
      testErr = testerr:value(1),
      testsinglecrop = config.testsinglecrop and 1 or 0
   }
   log:flush()
end

io.write('| training the network...\n')
engine:train{
   network   = classifier,
   criterion = crit,
   iterator  = train,
   lr        = config.lr,
   maxepoch  = config.maxepoch
}
