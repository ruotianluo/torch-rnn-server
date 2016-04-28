require 'torch'
require 'nn'
require 'LanguageModel'

local cmd = torch.CmdLine()
cmd:option('-checkpoint', '/home/robin/dev/checkpoints/juliet_70000.t7')
cmd:option('-word_limit_short', 3)
cmd:option('-word_limit_long', 20)
cmd:option('-port', 8080)
cmd:option('-gpu', 0)
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local msg
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
else
  print('Running in CPU mode... it will be slow!')
end

model:evaluate()

-- utility

-- http://lua-users.org/wiki/StringRecipes
function url_decode(str)
  str = string.gsub(str, '+', ' ')
  str = string.gsub(str, '%%(%x%x)', function(h) return string.char(tonumber(h,16)) end)
  str = string.gsub(str, '\r\n', '\n')
  return str
end

-- let's do this

local app = require('waffle') -- wish i knew how to suppress the graphicsmagick error message… oh well

app.set('host', '0.0.0.0')
app.set('port', opt.port)
app.set('debug', true)

app.get('/', function(req, res)
  res.json{message='Nobody here but us chickens!'}
end)

app.get('/generate', function(req, res)
  local start_text = req.url.args.start_text
  if start_text ~= nil then -- lua's 'not equals' is weird
    start_text = url_decode(start_text:lower():gsub('\n',''))
  else
    start_text = ''
  end

  local target_num = tonumber(req.url.args.n)
  if target_num == nil then target_num = 1 end

  local skip_num = 0
  local t0 = os.clock()

  local generated = {}

  while #generated < target_num do
    local sentence = model:sample(start_text, '[!?\\.]', 3)
    local num_words = 0
    for word in (sentence..' '):gmatch('(.-) ') do num_words = num_words + 1 end -- a bit of a mess
    if (num_words >= opt.word_limit_short) and (num_words <= opt.word_limit_long) then
      table.insert(generated, sentence)
    else
      skip_num = skip_num + 1
    end
  end

  print('Generated ' .. #generated .. ' sentences from start text:')
  print('  ' .. start_text)
  print('Skipped ' .. skip_num .. ' sentences')
  print('Took ' .. string.format('%.2f', os.clock() - t0) .. ' seconds')

  res.json{sentences=generated, time=elapsed}
end)

app.listen()