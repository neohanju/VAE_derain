
local torch = require 'torch'
local util = paths.dofile('util.lua')
local display = require 'display'  -- need to 'luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec' for display through browser

-- utils.loadImagePairs(0)

local realImage, synthImage = utils.loadImagePairs(0, 0)
-- display.image(realImage)
-- display.image(synthImage)
local imageCat = util.imageCatH(realImage, synthImage)
display.image(imageCat)


-- ()()
-- ('')HAANJU.YOO
