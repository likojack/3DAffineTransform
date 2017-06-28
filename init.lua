require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libstntd'
if withCuda then
   require 'libcustntd'
end

require('stntd.AffineGridGeneratorThreeD')
require('stntd.BilinearSamplerThreeD')
require('stntd.AffineTransformMatrixGenerator')
require('stntd.QuatToEuler')
require('stntd.CustomSoftMaxCriterion')
require('stntd.SE3TransformMatrix')


return nn
