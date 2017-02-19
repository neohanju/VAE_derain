-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'

--=============================================================================
-- PARAMETERS (and parsing)
--=============================================================================
local cmd = torch.CmdLine()
cmd:option('-datasetPath', '',      'path to dataset')
cmd:option('-outputPath',  '',      'path to saving result folder')
cmd:option('-model', 'VAE',         'model to use: ConvAE, DenoisingAE, UpconvAE, VAE')
cmd:option('-learningRate', 0.0001, 'Learning rate')
cmd:option('-epochs', 20,           'Training epochs')
cmd:option('-batchSize', 100,       '# of images in batch')

local opt = cmd:parse(arg)

-- Set up Torch
print('Setting up Torch')
opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

local cuda = pcall(require, 'cutorch') -- Use CUDA if available
if cuda then
  require 'cunn'
  cutorch.manualSeed(opt.manualSeed)
end


--=============================================================================
-- NETWORK
--=============================================================================

-- load data


-- create model
local Model = require ('models/' .. opt.model)





-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

-- channel index (for access feed data for discriminator)
if opt.which_direction=='AtoB' then
	idx_A = {1, input_nc}
	idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
	idx_A = {input_nc+1, input_nc+output_nc}
	idx_B = {1, input_nc}
else
	error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end
if opt.display then disp = require 'display' end



torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()


-- Set up Torch
print('Setting up Torch')
opt.manualSeed = torch.random(1, 10000) -- fix seed
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')
print("Random Seed: " .. opt.manualSeed)
if cuda then
  require 'cunn'
  cutorch.manualSeed(torch.random())
end


--=============================================================================
-- NETWORK RELATED
--=============================================================================
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local function weights_init(m)
	local name = torch.type(m)
	if name:find('Convolution') then
		m.weight:normal(0.0, 0.02)
		m.bias:fill(0)
	elseif name:find('BatchNormalization') then
		if m.weight then m.weight:normal(1.0, 0.02) end
		if m.bias then m.bias:fill(0) end
	end
end

function defineG(input_nc, output_nc, ngf)
	local netG = nil

	if opt.which_model_netG == "encoder_decoder" then
		netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
	elseif opt.which_model_netG == "unet" then
		netG = defineG_unet(input_nc, output_nc, ngf)
	elseif opt.which_model_netG == "unet_128" then 
		netG = defineG_unet_128(input_nc, output_nc, ngf)
	else 
		error("unsupported netG model")
	end

	netG:apply(weights_init)
	return netG
end

function defineD(input_nc, output_nc, ndf)
	local netD = nil
	if opt.condition_GAN==1 then
		input_nc_tmp = input_nc
	else
		input_nc_tmp = 0 -- only penalizes structure in output channels
	end

	if opt.which_model_netD == "basic" then
		netD = defineD_basic(input_nc_tmp, output_nc, ndf)
	elseif opt.which_model_netD == "n_layers" then
		netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
	else
		error("unsupported netD model")
	end

	netD:apply(weights_init)
	return netD
end

-- load saved models and finetune
if opt.continue_train == 1 then
	print('loading previously trained netG...')
	netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
	print('loading previously trained netD...')
	netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
else
	print('define model netG...')
	netG = defineG(input_nc, output_nc, ngf)
	print('define model netD...')
	netD = defineD(input_nc, output_nc, ndf)
end
print(netG)
print(netD)


local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local real_A  = torch.Tensor(opt.batchSize, input_nc,  opt.fineSize, opt.fineSize)
local real_B  = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B  = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

-- for GPU
if opt.gpu > 0 then
	print('transferring to gpu...')
	require 'cunn'
	cutorch.setDevice(opt.gpu)
	real_A  = real_A:cuda();
	real_B  = real_B:cuda();  fake_B  = fake_B:cuda();
	real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
	if opt.cudnn==1 then
		netG = util.cudnn(netG); netD = util.cudnn(netD);
	end
	netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();
	print('done')
else
	print('running model on CPU')
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()


--=============================================================================
-- IMAGE GENERATION
--=============================================================================
function createRealFake()
	--[[
	In case of conditional GAN ...
		real_AB = real image + real image
		fake_AB = real image + generated image
	but, for unconditional GAN ...
		real_AB = real image
		fake_AB = generated image
	--]]

	-- load real
	data_tm:reset(); data_tm:resume()
	local real_data, data_path = data:getBatch()
	data_tm:stop()

	real_A:copy(real_data[{ {}, idx_A, {}, {} }])
	real_B:copy(real_data[{ {}, idx_B, {}, {} }])

	if opt.condition_GAN == 1 then
		real_AB = torch.cat(real_A, real_B, 2)
	else
		real_AB = real_B  -- unconditional GAN, only penalizes structure in B
	end

	-- create fake
	fake_B = netG:forward(real_A)

	if opt.condition_GAN == 1 then
		fake_AB = torch.cat(real_A, fake_B, 2)
	else
		fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
	end
	local predict_real = netD:forward(real_AB)
	local predict_fake = netD:forward(fake_AB)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
	netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
	netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

	gradParametersD:zero()  -- I cannot any idea why this variable is needed

	-- Real
	local output = netD:forward(real_AB)
	local label  = torch.FloatTensor(output:size()):fill(real_label)
	if opt.gpu > 0 then 
		label = label:cuda()
	end

	local errD_real = criterion:forward(output, label)
	local df_do     = criterion:backward(output, label)
	netD:backward(real_AB, df_do)

	-- Fake
	local output = netD:forward(fake_AB)
	label:fill(fake_label)

	local errD_fake = criterion:forward(output, label)
	local df_do     = criterion:backward(output, label)
	netD:backward(fake_AB, df_do)

	-- total error of discriminator
	errD = (errD_real + errD_fake) / 2

	return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
	netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
	netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

	gradParametersG:zero()

	-- GAN loss
	local df_dg = torch.zeros(fake_B:size())
	if opt.gpu > 0 then 
		df_dg = df_dg:cuda();
	end

	if opt.use_GAN==1 then
		local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
		local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
		if opt.gpu > 0 then 
			label = label:cuda();
			end
		errG = criterion:forward(output, label)
		local df_do = criterion:backward(output, label)
		df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
	else
		errG = 0
	end

	-- unary loss
	local df_do_AE = torch.zeros(fake_B:size())
	if opt.gpu>0 then 
		df_do_AE = df_do_AE:cuda();
	end
	if opt.use_L1==1 then
		errL1 = criterionAE:forward(fake_B, real_B)
		df_do_AE = criterionAE:backward(fake_B, real_B)
	else
		errL1 = 0
	end

	netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))

	return errG, gradParametersG
end
