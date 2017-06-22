require 'stntd'
require 'nn'
local matio = require 'matio'

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

params = torch.Tensor(1,9):fill(0)

params[1][4] = 2
params[1][5] = 1
params[1][6] = 1
params[1][7] = -1 -- z
params[1][8] = 0 -- y
params[1][9] = 0
matrixGenerator = nn.AffineTransformMatrixGenerator(true,true,true)
transMatrix = matrixGenerator:updateOutput(params)
gridSpace = 5
gridGenerator = nn.AffineGridGeneratorThreeD(gridSpace,gridSpace,gridSpace)
transGrid = gridGenerator:updateOutput(transMatrix)
voxelGrid = torch.Tensor(1,3,3,3,1):fill(1)
sampler = nn.BilinearSamplerThreeD()
outSDF = sampler:updateOutput({voxelGrid, transGrid})
print(outSDF:select(1,1):select(4,1))
print(transGrid:select(1,1):select(4,1))
