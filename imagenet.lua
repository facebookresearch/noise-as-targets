[[--
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.
--]]

local tnt       = require 'torchnet'
local transform = require 'torchnet.transform'
local image = require 'image'

local unsup = {}
local data = {}

local ImagenetDataset, parent = torch.class(
   'unsup.ImagenetDataset',
   'tnt.TransformDataset',
   unsup
)

function data.randomcrop(isz, lisz)
   local image = require 'image'
   return
      function(img)
         -- short side is lisz
         img = image.scale(img, "^" .. math.max(isz, lisz))

         -- random crop
         local iw, ih = img:size(3), img:size(2)
         local sx = torch.random(0, iw-isz)
         local sy = torch.random(0, ih-isz)
         img = image.crop(img, sx, sy, sx+isz, sy+isz)

         -- random flip
         if torch.uniform() > 0.5 then
            img = image.hflip(img)
         end

         return img
      end
end

function data.crop(isz, lisz, format, flip)
   local image = require 'image'
   return
      function(img)
         -- short side is lisz
         img = image.scale(img, "^" .. math.max(isz, lisz))

         -- flip if needed
         if flip then
            img = image.hflip(img)
         end

         -- crop
         img = image.crop(img, format, isz, isz)

         return img
      end
end

function ImagenetDataset:__init(config, split, mode, maxload)
   local isz, lisz = config.cropsize, config.imsize
   local split = split or 'train'
   if split == 'test' or split == 'val' then
      split = 'valid'
   end
   local mode = mode or split
   local trans
   if mode == 'train' then
      trans = transform.compose{
         function(b) return image.decompress(b, 3, 'float') end,
         data.randomcrop(isz, lisz),
         transform.normalize(),
      }
   else
      trans = transform.compose{
         function(b) return image.decompress(b, 3, 'float') end,
         config.testsinglecrop and data.crop(isz, lisz, 'c',  false)
         or transform.merge{
           data.crop(isz, lisz, 'c',  false),
           data.crop(isz, lisz, 'tl', false),
           data.crop(isz, lisz, 'tr', false),
           data.crop(isz, lisz, 'bl', false),
           data.crop(isz, lisz, 'br', false),
           data.crop(isz, lisz, 'c',  true),
           data.crop(isz, lisz, 'tl', true),
           data.crop(isz, lisz, 'tr', true),
           data.crop(isz, lisz, 'bl', true),
           data.crop(isz, lisz, 'br', true)
         },
         transform.normalize(),
      }
   end
   parent.__init(
      self,
      tnt.IndexedDataset{
         path    = string.format('%s/%s', config.datadir, split),
         fields  = {'input', 'target'},
         maxload = config.maxload,
         mmap    = false,
         mmapidx = true,
      },
      trans,
      'input'
   )
end

return unsup.ImagenetDataset
