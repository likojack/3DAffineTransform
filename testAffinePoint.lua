require 'nn'
require 'stntd'

transform = dofile("AffineTransformPoint.lua")
transformPoint = transform.AffineTransformPoint()
martrixGen = nn.AffineTransformMatrixGenerator(true, true, true)
transParams = torch.Tensor(1,9):fill(0)
transParams[{{},{4,-1}}] = 1
transMatrix = martrixGen:updateOutput(transParams)
nPoint = 5
pointList = torch.Tensor(nPoint, 3):random(1,3)
if(pointList:size()[2] == 3) then
  local homo = torch.Tensor(nPoint,1):fill(1)
  pointList = torch.cat(pointList, homo, 2)
end
print(pointList) 
transMatrix = transMatrix[1]
out = transformPoint:forward({pointList, transMatrix})
gradOut = out:clone():fill(1)
gradIn = transformPoint:backward({pointList, transMatrix}, gradOut)
print(gradIn[2])
-- out = transform.AffineTransformPoint:updateOutput({pointList, transMatrix})
-- print(pointList)
-- print(out)
-- gradOut = pointList:clone():fill(1)
-- gradinput = transform:updateGradInput({pointList, transformMatrix}, gradOut)
-- print(transformMatrix:transpose(2,1))
-- print(pointList)
-- print(out)
-- print(gradinput[1])


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