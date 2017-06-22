require 'nn'
require 'stntd'

transform = nn.AffineTransformPoint()
transformMatrix = torch.Tensor(3,4):range(1,12)
-- transformMatrix[1][1] = 1
-- transformMatrix[2][2] = 1
-- transformMatrix[3][3] = 1
nPoint = 5
pointList = torch.Tensor(nPoint, 3):random(1,3)
out = transform:updateOutput({pointList, transformMatrix})
gradOut = pointList:clone():fill(1)
gradinput = transform:updateGradInput({pointList, transformMatrix}, gradOut)
print(transformMatrix:transpose(2,1))
print(pointList)
print(out)
print(gradinput[1])

-- local module = nn.AffineTransformPoint()
-- module._updateOutput = module.updateOutput
-- function module:updateOutput(input)
--    return self:_updateOutput({input, transformMatrix})
-- end
-- module._updateGradInput = module.updateGradInput
-- function module:updateGradInput(input,gradOutput)
--    self:_updateGradInput({input, transformMatrix}, gradOutput)
--    return self.gradInput[1]
-- end
-- local jac = nn.Jacobian
-- local precision = 1e-7
-- local err = jac.testJacobian(module,pointList)
-- print('==> error point: ' .. err)
-- if err<precision then
--    print('==> module OK')
-- else
--       print('==> error too large, incorrect implementation')
-- end

-- module._updateOutput = module.updateOutput
-- function module:updateOutput(input)
--    return self:_updateOutput({pointList, input})
-- end
-- module._updateGradInput = module.updateGradInput
-- function module:updateGradInput(input,gradOutput)
--    self:_updateGradInput({pointList, input}, gradOutput)
--    return self.gradInput[2]
-- end
-- local jac = nn.Jacobian
-- local precision = 1e-7
-- local err = jac.testJacobian(module,transformMatrix)
-- print('==> error matrix: ' .. err)
-- if err<precision then
--    print('==> module OK')
-- else
--       print('==> error too large, incorrect implementation')
-- end