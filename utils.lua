require 'torch'
require 'image'

local utils = {}

-- PRESETS
local DATASET_BASE_DIR = "/home/neohanju/Workspace/dataset"
local DATASET_NAME     = "HJ_Rain"
local DATASET_ORIGINAL = "original"
local DATASET_SYNTH    = "synthesized"
local NUM_CHANNEL = 3

function utils.loadImagePairs(imageIndex, synthsizedIndex)
	
	local originalImagePath = string.format("%s/%s/%s/%06d.jpg", 
		DATASET_BASE_DIR, DATASET_NAME, DATASET_ORIGINAL, imageIndex)
	
	local synthesizedImagePath = string.format("%s/%s/%s/%06d_rain_%04d.jpg", 
		DATASET_BASE_DIR, DATASET_NAME, DATASET_SYNTH, imageIndex, synthsizedIndex)

	-- local originalImage = image.load(originalImagePath, NUM_CHANNEL, 'byte')

	return image.load(originalImagePath, NUM_CHANNEL, 'byte'), 
		image.load(synthesizedImagePath, NUM_CHANNEL, 'byte')
end

return utils

-- ()()
-- ('')HAANJU.YOO
