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
require 'cudnn'
local tnt = require 'torchnet'
local paths = require 'paths'
local hungarian = require 'hungarian'

cudnn.fastest   = true
cudnn.benchmark = true

-- load some options from the command line:
local cmd = torch.CmdLine()
cmd:option('-dataroot', '', 'path to folder containing imagenet-idx')
cmd:option('-rundir', '', 'path to experiment root directory')

cmd:option('-nthread', 10, 'number of dataset iterator threads')
cmd:option('-gpu', 1, 'id of GPU to use')
cmd:option('-ngpu', 1, 'number of GPUs to use in //')
cmd:option('-seed', 1111, 'random seed')

cmd:option('-batch', 256, 'batch size')
cmd:option('-maxepoch', 100, 'number of training epochs')
cmd:option('-maxload', -1, 'number of training examples')
cmd:option('-saveperiod', 10, 'period for saving models')

cmd:option('-lr', 0.01, 'learning rate')
cmd:option('-dim', 2048, 'size of the output layer')
cmd:option('-permute', 3, 'period for permuting targets')
local config = cmd:parse(arg)
assert(config.rundir  ~= '', 'Please provide a path for results!')
assert(config.dataroot  ~= '', 'Please provide a path for data!')

-- various initializations:
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)
torch.manualSeed(config.seed)

config.datadir = string.format('%s/imagenet-idx', config.dataroot)
config.loader = 'imagenet'
config.imsize  = 256
config.cropsize = 224
config.arch = 'alexnet'
config.specs = 'alexnet-specs'
config.nclass = config.dim

if config.maxload == -1 then config.maxload = nil end

-- set running dir and save the model:
tnt.utils.sys.mkdir(config.rundir)
local rundir = string.format(
   '%s/unsup-dim-%d-perm-%d-lr-%3.1e',
   config.rundir,
   config.dim,
   config.permute,
   config.lr
)
tnt.utils.sys.mkdir(rundir)
torch.save(string.format('%s/config.bin', rundir), config)
io.write(string.format('| Running in directory %s\n', rundir))

-- get the size of the dataset
local tempData = require(config.loader)(config, 'train')
local nimg = tempData:size()
config.ncluster = nimg
io.write(string.format('| Found %d images.\n', nimg))

-- get the network architecture
local arch = require(string.format("%s", config.arch))

-- either load everything from checkpoint or allocate new
local net, codes, crit, label, checkpoint
local checkpointFname = string.format('%s/checkpoint.bin', rundir)
if paths.filep(checkpointFname) then
   io.write(string.format('| Found checkpoint at %s.\n', checkpointFname))
   checkpoint = torch.load(checkpointFname)
   codes = checkpoint.codes
   crit = checkpoint.crit:cuda()
   label = checkpoint.label
   net = arch.loadNamed(rundir, 'checkpointModel'):cuda()
   config.seed = checkpoint.seed + 1
   torch.manualSeed(config.seed)
else
   io.write('| Getting the model...\n')
   net  = arch.train(config):cuda()
   io.write('| Creating the filter...\n')
   local filter
   filter = cudnn.SpatialConvolution(3, 2, 3, 3, 1, 1, 1, 1)
   local dx = (1.0 / 3.0) * torch.FloatTensor(
      {
         {-1.0, 0.0, 1.0},
         {-2.0, 0.0, 2.0},
         {-1.0, 0.0, 1.0}
      }
   )
   local dy = (1.0 / 3.0) * torch.FloatTensor(
      {
         {-1.0, -2.0, -1.0},
         {0.0, 0.0, 0.0},
         {1.0, 2.0, 1.0}
      }
   )
   filter.weight[1][1] = dx
   filter.weight[1][2] = dx
   filter.weight[1][3] = dx
   filter.weight[2][1] = dy
   filter.weight[2][2] = dy
   filter.weight[2][3] = dy
   filter = filter:cuda()
   net:insert(filter, 1)
   io.write('| Generating the targets...\n')
   codes = torch.randn(config.ncluster, config.nclass)
   for i = 1, codes:size(1) do
      codes[i]:mul(1.0 / torch.norm(codes[i]))
   end
   crit = nn.MSECriterion():cuda()
   crit.sizeAverage = false
   net:add(nn.Normalize(2))
   net = net:cuda()
   label = torch.mod(torch.randperm(nimg), config.ncluster):add(1)
   crit = crit:cuda()
end

-- allocating containers for batch and batch score
local input, target = torch.CudaTensor(), torch.CudaTensor()
local scpu = torch.DoubleTensor(config.batch, config.batch)
local s = torch.CudaTensor(config.batch, config.batch):zero()

-- make training parallel
if config.ngpu > 1 then net = arch.parallelize(net,config) end

-- create dataset iterator
io.write('| Spawning data threads...\n')
local iterator = tnt.ParallelDatasetIterator{
   nthread = config.nthread,
   init = function()
      require 'cutorch'
      torch.manualSeed(config.seed)
   end,
   closure = function()
      local tnt       = require 'torchnet'
      return tnt.BatchDataset{
         dataset = tnt.ShuffleDataset{
            dataset = tnt.TransformDataset{
               dataset = require(config.loader)(config),
               transform = function(sample, idx)
                  sample["class"] = sample.target
                  if config.geometry ~= 'supervised' then
                     sample["target"] = codes[label[idx]]
                  end
                  sample["index"] = torch.LongTensor({idx})
                  return sample
               end,
            },
            replacement = false,
         },
         batchsize = config.batch,
         policy = 'skip-last',
      }
   end,
   ordered = true
}
local trainsetsize = iterator:execSingle('size')
io.write(string.format('| There are %d batches in the training set\n', trainsetsize))

-- set up learner
local engine = tnt.SGDEngine()

-- measure time and loss
local timer = tnt.TimeMeter{unit = true}
local loss = tnt.AverageValueMeter()

-- setup logger
local logtext = require 'torchnet.log.view.json'
local logkeys = {"permute", "lr", "epoch", "loss"}
local logformat = {"%d", "%4.2e", "%d", "%6.4f"}
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

-- reset all the meters
function engine.hooks.onStartEpoch(state)
   loss:reset()
   timer:reset()
   timer:resume()
   iterator:exec('resample')
   state.t = 0
   if checkpoint and state.epoch < checkpoint.epoch then
      state.epoch = checkpoint.epoch
   end
end

function engine.hooks.onSample(state)
   -- move the data to GPU
   if type(state.sample.target) == 'table' then
      state.sample.target = torch.Tensor(state.sample.target)
   end
   input:resize( state.sample.input:size() ):copy(state.sample.input)
   target:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input  = input
   state.sample.target = target:squeeze()
end

function engine.hooks.onForward(state)
   -- every other config.permute epochs, permute the targets
   if state.epoch % config.permute == 0 then
      local z = state.network.output
      local code = state.sample.target
      -- computing score inside the batch
      torch.mm(s, z, code:t())
      scpu:copy(s)

      local assignment = hungarian.maxCost(scpu)

      -- update the targets
      state.sample.target = code:index(1, assignment:squeeze():long())
      local indices = state.sample.index:squeeze():long()
      local oldLabel = label:index(1, indices)
      local newLabel = oldLabel:index(1, assignment:long())
      label:indexCopy(1, indices, newLabel)
   end
end

function engine.hooks.onForwardCriterion(state)
   loss:add(state.criterion.output / config.batch)
   timer:incUnit()
end

function engine.hooks.onBackward(state)
   -- zero-out the gradients for the sobel filter
   if torch.type(net) == 'nn.DataParallel'
         or torch.type(net) == 'nn.DataParallelTable' then
      state.network:get(1):get(1):zeroGradParameters()
   else
      state.network:get(1):zeroGradParameters()
   end
end

function engine.hooks.onUpdate(state)
   if state.t % 10 == 0 then
      io.write(
         string.format(
            '\tepoch=%-5d batch=%-5d/%-5d loss=%-7.5f ms/b=%-7.0f\n',
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

-- perform testing and print out progress:
function engine.hooks.onEndEpoch(state)
   timer:stop()
   local curloss = loss:value()

   -- write log
   log:set{
      permute = config.permute,
      lr = state.lr,
      epoch = state.epoch,
      loss = loss:value(),
   }
   log:flush()

   -- create checkpoint table
   local toCheckpoint = {}
   toCheckpoint.codes = codes
   toCheckpoint.label = label
   toCheckpoint.crit = crit
   toCheckpoint.epoch = state.epoch
   toCheckpoint.seed = config.seed

   -- save checkpoint in a rolling fashion
   -- set symlinks to latest
   local rollingCheckpointFname = string.format(
      '%s/rcp-%d-checkpoint.bin',
      rundir,
      state.epoch % 3
   )
   torch.save(rollingCheckpointFname, toCheckpoint)
   arch.saveNamed(
      net,
      rundir,
      string.format('rcp-%d-checkpointModel', state.epoch % 3)
   )
   os.execute(string.format(
      'ln -s -f %s %s',
      string.format('./rcp-%d-checkpoint.bin', state.epoch % 3),
      string.format('%s/checkpoint.bin', rundir)
   ))
   os.execute(string.format(
      'ln -s -f %s %s',
      string.format('./rcp-%d-checkpointModel.bin', state.epoch % 3),
      string.format('%s/checkpointModel.bin', rundir)
   ))
   if state.epoch % config.saveperiod == 0 then
      local savedir = string.format('%s/epoch-%05d', rundir, state.epoch)
      tnt.utils.sys.mkdir(savedir)
      arch.saveNamed(net, savedir, 'model')
      local backupFname = string.format('%s/%s', savedir, 'checkpoint.bin')
      torch.save(backupFname, toCheckpoint)
   end
end

io.write('| training the network...\n')
engine:train{
   network = net,
   criterion = crit,
   iterator = iterator,
   lr = config.lr,
   maxepoch = config.maxepoch
}
