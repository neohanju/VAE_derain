local hj_load_data = {}

require 'image'

function hj_load_data.read(path)
	local image = image.load(path, 3, 'float')
	return image
end

-- augment image with reflection and cropping
function hj_load_data.augmentation(image, output_size)
	local num_channels, height, width = image:size(1), image:size(2), image:size(3)
end
