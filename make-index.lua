[[--
  Copyright (c) 2017-present, Facebook, Inc.
  All rights reserved.

  This source code is licensed under the BSD-style license found in the
  LICENSE file in the root directory of this source tree. An additional grant
  of patent rights can be found in the PATENTS file in the same directory.
--]]

require 'torch'
require 'lfs'
local tnt = require 'torchnet'

torch.manualSeed(1111)

local function fileapply(path, regex, closure)
   for filename in lfs.dir(path) do
      if filename ~= '.' and filename ~= '..' then
         filename = path .. '/' .. filename
         if lfs.attributes(filename, 'mode') == 'directory' then
            fileapply(filename, regex, closure)
         else
            if not regex or filename:match(regex) then
               closure(filename)
            end
         end
      end
   end
end

assert(arg[1],
       string.format(
          'usage: %s <path to imagenet> <output directory>',
          arg[0]))

assert(arg[2],
       string.format(
          'usage: %s <path to imagenet> <output directory>',
          arg[0]))

assert(
   os.execute(string.format('mkdir -p %s', arg[2])),
   'could not create output directory')

local function filename2classname(filename)
   local classname = filename:match('/n(%d+)/')
   assert(classname, string.format('invalid filename <%s>', filename))
   return classname
end

local function isimage(filename)
   local ext = filename:match('[^%.]+$')
   if ext then
      ext = ext:lower()
   end
   return (ext == 'jpeg' or ext == 'jpg' or ext == 'png')
end

local function writeidx(subpaths)
   local classes = subpaths.classes
   if not classes then
      print(string.format("| finding all classes in %s...", arg[1] .. '/' .. subpaths.src))
      local uoclasses = {}
      fileapply(
         arg[1] .. '/' .. subpaths.src,
         nil,
         function(filename)
            if isimage(filename) then
               local classname = filename2classname(filename)
               if not uoclasses[classname] then
                  table.insert(uoclasses, classname)
                  uoclasses[classname] = #uoclasses
               end
            end
         end
      )
      table.sort(
         uoclasses,
         function(a, b)
            return a < b
         end
      )
      classes = {}
      for id, name in ipairs(uoclasses) do
         classes[id] = name
         classes[name] = id
      end
      print(string.format("| %d classes found", #classes))
   end

   print(string.format("| analyzing %s...", arg[1] .. '/' .. subpaths.src))
   local filelst = {}
   fileapply(
      arg[1] .. '/' .. subpaths.src,
      nil,
      function(filename)
         if isimage(filename) then
            table.insert(filelst, filename)
         end
      end
   )

   local root = string.format("%s/%s", arg[2], subpaths.dst)
   assert(
      os.execute(string.format('mkdir -p %s', root)),
      'could not create output directory')

   local inputidx = tnt.IndexedDatasetWriter(
      string.format("%s/input.idx", root),
      string.format("%s/input.bin", root),
      "byte"
   )
   local targetidx = tnt.IndexedDatasetWriter(
      string.format("%s/target.idx", root),
      string.format("%s/target.bin", root),
      "long"
   )

   print("| writing the index...")
   local perm = torch.randperm(#filelst) -- we shuffle examples
   local nclass = #classes
   for i=1,#filelst do
      local filename = filelst[perm[i]]
      local classid = filename2classname(filename)
      classid = classes[classid]
      assert(classid and classid > 0 and classid <= nclass)
      inputidx:add(filename)
      targetidx:add(torch.LongTensor(1):fill(classid))
   end
   inputidx:close()
   targetidx:close()
   print("| done.")

   return classes
end

local classes = writeidx{src='train', dst='train'}
writeidx{src='val', dst='valid', classes=classes}

print(string.format("| writing the %d classes hash", #classes))
local f = io.open(string.format("%s/class.lst", arg[2]), 'w')
for id,classname in ipairs(classes) do
   f:write(classname .. '\n')
end
f:close()
