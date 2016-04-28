require 'torch'
require 'nn'

require 'LanguageModel'

local cmd = torch.CmdLine()
cmd:option('-checkpoint', '/home/robin/dev/checkpoints/juliet_70000.t7')
cmd:option('-start_text', '')
cmd:option('-gpu', 0)
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

-- note: i don't use this script anymore; might be funky

local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end

print(msg)

model:evaluate()

local sample = model:sample(opt.start_text, '[!?\\.]', 3)
print(sample)