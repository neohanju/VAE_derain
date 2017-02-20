require 'nngraph'
require '../modules/Gaussian'

local Model = {
  zSize = 100 -- Size of isotropic multivariate Gaussian Z
}

function Model:createAutoencoder(X)
  local netG = nil
  -- input is (nc) x 256 x 256
  local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
  -- input is (ngf) x 128 x 128
  local e2 = e1 
            - nn.LeakyReLU(0.2, true) 
            - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) 
            - nn.SpatialBatchNormalization(ngf * 2)
  -- input is (ngf * 2) x 64 x 64
  local e3 = e2 
            - nn.LeakyReLU(0.2, true) 
            - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) 
            - nn.SpatialBatchNormalization(ngf * 4)
  -- input is (ngf * 4) x 32 x 32
  local e4 = e3 
            - nn.LeakyReLU(0.2, true) 
            - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) 
            - nn.SpatialBatchNormalization(ngf * 8)
  -- input is (ngf * 8) x 16 x 16
  local e5 = e4 
            - nn.LeakyReLU(0.2, true) 
            - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) 
            - nn.SpatialBatchNormalization(ngf * 8)
  -- input is (ngf * 8) x 8 x 8
  local e6 = e5 
            - nn.LeakyReLU(0.2, true) 
            - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) 
            - nn.SpatialBatchNormalization(ngf * 8)
  -- input is (ngf * 8) x 4 x 4
  local e7 = e6 
            - nn.LeakyReLU(0.2, true) 
            - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) 
            - nn.SpatialBatchNormalization(ngf * 8)
  -- input is (ngf * 8) x 2 x 2
  local e8 = e7 
            - nn.LeakyReLU(0.2, true) 
            - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) 
            - nn.SpatialBatchNormalization(ngf * 8)  

  --===========================================================================
  -- BELOW: added by neohanju (2017.02.20)
  -- Create latent Z parameter layer
  local zMean = e8 - nn.Linear(ngf * 8, self.zSize) -- Mean μ of Z
  local zCov  = e8 - nn.Linear(ngf * 8, self.zSize) -- Log variance σ^2 of Z (diagonal covariance)

  -- Create σε module
  local stdModule   = zCov 
                      - nn.MulConstant(0.5) -- Compute 1/2 log σ^2 = log σ
                      - nn.Exp()            -- Compute σ 
  local noiseModule = - {stdModule, nn.Gaussian(0, 1)} -- Sample noise ε ~ N(0, 1)
                      - nn.CMulTable()                 -- Compute σε

  -- Create sampler q(z) = N(z; μ, σI) = μ + σε (reparametrization trick)
  local sampler = {zMean, noiseModule} - nn.CAddTable()
  --===========================================================================
  -- local d1_ = e8
  --             - nn.ReLU(true) 
  --             - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) 
  --             - nn.SpatialBatchNormalization(ngf * 8) 
  --             - nn.Dropout(0.5)

  -- input is (ngf * 8) x 1 x 1
  local d1_ = sampler 
              - nn.ReLU(true) 
              - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) 
              - nn.SpatialBatchNormalization(ngf * 8) 
              - nn.Dropout(0.5)
  -- input is (ngf * 8) x 2 x 2
  local d1 = {d1_, e7} - nn.JoinTable(2)
  local d2_ = d1 
              - nn.ReLU(true) 
              - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) 
              - nn.SpatialBatchNormalization(ngf * 8) 
              - nn.Dropout(0.5)
  -- input is (ngf * 8) x 4 x 4
  local d2 = {d2_, e6} - nn.JoinTable(2)
  local d3_ = d2 
              - nn.ReLU(true) 
              - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) 
              - nn.SpatialBatchNormalization(ngf * 8) 
              - nn.Dropout(0.5)
  -- input is (ngf * 8) x 8 x 8
  local d3 = {d3_, e5} - nn.JoinTable(2)
  local d4_ = d3 
              - nn.ReLU(true) 
              - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) 
              - nn.SpatialBatchNormalization(ngf * 8)
  -- input is (ngf * 8) x 16 x 16
  local d4 = {d4_, e4} - nn.JoinTable(2)
  local d5_ = d4 
              - nn.ReLU(true) 
              - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) 
              - nn.SpatialBatchNormalization(ngf * 4)
  -- input is (ngf * 4) x 32 x 32
  local d5 = {d5_, e3} - nn.JoinTable(2)
  local d6_ = d5 
              - nn.ReLU(true) 
              - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) 
              - nn.SpatialBatchNormalization(ngf * 2)
  -- input is (ngf * 2) x 64 x 64
  local d6 = {d6_, e2} - nn.JoinTable(2)
  local d7_ = d6 
              - nn.ReLU(true) 
              - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) 
              - nn.SpatialBatchNormalization(ngf)
  -- input is (ngf) x128 x 128
  local d7 = {d7_, e1} - nn.JoinTable(2)
  local d8 = d7 
              - nn.ReLU(true) 
              - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
  -- input is (nc) x 256 x 256

  local o1 = d8 - nn.Tanh()

  netG = nn.gModule({e1}, {o1})

  --graph.dot(netG.fg,'netG')

  return netG


  -- input is (ngf * 8) x 1 x 1






  -- Create latent Z parameter layer
  local zLayer = nn.ConcatTable()
  zLayer:add(nn.Linear(64, self.zSize)) -- Mean μ of Z
  zLayer:add(nn.Linear(64, self.zSize)) -- Log variance σ^2 of Z (diagonal covariance)
  self.encoder:add(zLayer) -- Add Z parameter layer

  -- Create σε module
  local noiseModule = nn.Sequential()
  local noiseModuleInternal = nn.ConcatTable()
  local stdModule = nn.Sequential()
  stdModule:add(nn.MulConstant(0.5)) -- Compute 1/2 log σ^2 = log σ
  stdModule:add(nn.Exp()) -- Compute σ
  noiseModuleInternal:add(stdModule) -- Standard deviation σ
  noiseModuleInternal:add(nn.Gaussian(0, 1)) -- Sample noise ε ~ N(0, 1)
  noiseModule:add(noiseModuleInternal)
  noiseModule:add(nn.CMulTable()) -- Compute σε

  -- Create sampler q(z) = N(z; μ, σI) = μ + σε (reparametrization trick)
  local sampler = nn.Sequential()
  local samplerInternal = nn.ParallelTable()
  samplerInternal:add(nn.Identity()) -- Pass through μ 
  samplerInternal:add(noiseModule) -- Create noise σ * ε
  sampler:add(samplerInternal)
  sampler:add(nn.CAddTable())

  -- Create decoder (generative model q)
  self.decoder = nn.Sequential()
  self.decoder.add(nn.LeakyReLU(0.2, true))
  self.decoder.add(nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1))
  self.decoder.add(nn.SpatialBatchNormalization(ngf * 8))
  self.decoder.add(nn.Dropout(0.5))



  self.decoder:add(nn.Linear(self.zSize, 64))
  self.decoder:add(nn.BatchNormalization(64))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(64, 128))
  self.decoder:add(nn.BatchNormalization(128))
  self.decoder:add(nn.ReLU(true))
  self.decoder:add(nn.Linear(128, featureSize))
  self.decoder:add(nn.Sigmoid(true))
  self.decoder:add(nn.View(X:size(2), X:size(3)))

  -- Create autoencoder
  self.autoencoder = nn.Sequential()
  self.autoencoder:add(self.encoder)
  self.autoencoder:add(sampler)
  self.autoencoder:add(self.decoder)
end

return Model


function defineG_unet(input_nc, output_nc, ngf)


  local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
  -- input is (ngf * 8) x 2 x 2
  local d1 = {d1_,e7} - nn.JoinTable(2)
  local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
  -- input is (ngf * 8) x 4 x 4
  local d2 = {d2_,e6} - nn.JoinTable(2)
  local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
  -- input is (ngf * 8) x 8 x 8
  local d3 = {d3_,e5} - nn.JoinTable(2)
  local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
  -- input is (ngf * 8) x 16 x 16
  local d4 = {d4_,e4} - nn.JoinTable(2)
  local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
  -- input is (ngf * 4) x 32 x 32
  local d5 = {d5_,e3} - nn.JoinTable(2)
  local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
  -- input is (ngf * 2) x 64 x 64
  local d6 = {d6_,e2} - nn.JoinTable(2)
  local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
  -- input is (ngf) x128 x 128
  local d7 = {d7_,e1} - nn.JoinTable(2)
  local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
  -- input is (nc) x 256 x 256

  local o1 = d8 - nn.Tanh()

  netG = nn.gModule({e1},{o1})

  --graph.dot(netG.fg,'netG')

  return netG
end