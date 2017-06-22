require 'nn'
require 'stntd'
-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- --test quat to euler
-- local module = nn.QuatToEuler()
-- local batchSize = torch.random(2,10)
-- local input = torch.zeros(batchSize,4):uniform()
-- local err = jac.testJacobian(module,input)
-- print('==> error quat: ' .. err)
-- if err<precision then
--    print('==> module OK')
-- else
--       print('==> error too large, incorrect implementation')
-- end



-- -- test matrix generator
-- useRotation = true
-- useScale = true
-- useTranslation = true
-- local module = nn.AffineTransformMatrixGenerator(useRotation,useScale,useTranslation)
-- nbNeededParams = 9
-- local nframes = torch.random(2,10)
-- local params = torch.zeros(nframes,nbNeededParams):uniform()
-- local err = jac.testJacobian(module,params)
-- print('==> error matrix: ' .. err)
-- if err<precision then
--    print('==> module OK')
-- else
--       print('==> error too large, incorrect implementation')
-- end


-- -- test grid generator
-- local nframes = torch.random(2,10)
-- local height = torch.random(2,5)
-- local width = torch.random(2,5)
-- local depth = torch.random(2,5)
-- local input = torch.zeros(nframes, 3, 4):uniform()
-- local module = nn.AffineGridGeneratorThreeD(depth, height, width)

-- local err = jac.testJacobian(module,input)
-- print('==> error: ' .. err)
-- if err<precision then
--    print('==> module OK')
-- else
--       print('==> error too large, incorrect implementation')
-- end


-- test sampler
local height = torch.random(1,5)
local width = torch.random(1,5)
local depth = torch.random(1,5)
local channels = torch.random(1,6)
local height = 8
local width = 8
local depth = 8
local channels = 1
local multiplier = 2
local inputImages = torch.zeros(1, depth, height, width, channels):uniform(0,1)
local grids = torch.zeros(1, depth*multiplier, height*multiplier, width*multiplier, 3):uniform(0, height-1)
local module = nn.BilinearSamplerThreeD()

-- test input images (first element of input table)
module._updateOutput = module.updateOutput
function module:updateOutput(input)
  return self:_updateOutput({input, grids})
end

module._updateGradInput = module.updateGradInput
function module:updateGradInput(input, gradOutput)
  self:_updateGradInput({input, grids}, gradOutput)
  return self.gradInput[1]
end

local errImages = jac.testJacobian(module,inputImages)
print('==> errorImage: ' .. errImages)
if errImages<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end

-- test grids (second element of input table)
function module:updateOutput(input)
  return self:_updateOutput({inputImages, input})
end

function module:updateGradInput(input, gradOutput)
  self:_updateGradInput({inputImages, input}, gradOutput)
  return self.gradInput[2]
end

local errGrids = jac.testJacobian(module,grids)
print('==> errorGrid: ' .. errGrids)
if errGrids<precision then
   print('==> module OK')
else
      print('==> error too large, incorrect implementation')
end


-- local function criterionJacobianTest(cri, input, target)
--    local eps = 1e-6
--    local _ = cri:forward(input, target)
--    local dfdx = cri:backward(input, target)
--    -- for each input perturbation, do central difference
--    local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
--    local input_s = input:storage()
--    local centraldiff_dfdx_s = centraldiff_dfdx:storage()
--    for i=1,input:nElement() do
--       -- f(xi + h)
--       input_s[i] = input_s[i] + eps

--       local fx1 = cri:forward(input, target)
--       -- f(xi - h)
--       input_s[i] = input_s[i] - 2*eps
--       local fx2 = cri:forward(input, target)
--       -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
--       local cdfx = (fx1 - fx2) / (2*eps)
--       -- store f' in appropriate place

--       centraldiff_dfdx_s[i] = cdfx

--       -- reset input[i]
--       input_s[i] = input_s[i] + eps
--    end
--    -- print(centraldiff_dfdx)
--    -- compare centraldiff_dfdx with :backward()
--    local err = (centraldiff_dfdx - dfdx):abs():max()
--    print(err)
--    -- mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
-- end

-- -- test criterion
-- local module = nn.CustomSoftMaxCriterion()
-- local eps = 1e-2
-- local input = torch.rand(10)*(1-eps) + eps/2
-- local target = torch.rand(10)*(1-eps) + eps/2
-- criterionJacobianTest(module, input, target)
-- -- print('==> error quat: ' .. err)
-- -- if err<precision then
-- --    print('==> module OK')
-- -- else
-- --       print('==> error too large, incorrect implementation')
-- -- end