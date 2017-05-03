require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libstntd'
if withCuda then
   require 'libcustntd'
end

--require('stn3d.AffineTransformMatrixGenerator')
require('stntd.AffineGridGeneratorThreeD')
require('stntd.BilinearSamplerThreeD')
--require('stn3d.TransformationMatrix3x4GeneratorEuler')

--require('stn.test')

return nn
