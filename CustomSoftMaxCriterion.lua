
local CustomSoftMaxCriterion, parent = torch.class('nn.CustomSoftMaxCriterion', 'nn.Criterion')

function CustomSoftMaxCriterion:__init(weights, sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
      self.sizeAverage = sizeAverage
   else
      self.sizeAverage = true
   end
   if weights ~= nil then
      assert(weights:dim() == 1, "weights input should be 1-D Tensor")
      self.weights = weights
	end
   self.weightedConstrainForward = 0
end

function CustomSoftMaxCriterion:updateOutput(input,target, primitiveSize, worldSize, constrainWeight)
   -- - log(input) * target - log(1 - input) * (1 - target)
   local outVol = input
   assert( outVol:nElement() == target:nElement(),
   "input and target size mismatch")
   local target = target
   local EPS = 1 * 10^(-12)
   local pixel_loss = - (torch.cmul(torch.log(outVol + EPS),target) + torch.cmul(torch.log(1 - outVol + EPS),(1 - target))) 
   local sum = torch.sum(pixel_loss)
   local avg_loss = sum / input:nElement()

   self.output = avg_loss
   return self.output
end


function CustomSoftMaxCriterion:updateGradInput(input, target,primitiveSize, worldSize)
   -- - (target - input) / ( input (1 - input) )
   local outVol = input
   local target = target
   local EPS = 1 * 10^(-12)
   assert( outVol:nElement() == target:nElement(),
   "input and target size mismatch")
   --backward wrt voxel
   self.gradInput = torch.cdiv((outVol - target), torch.cmul(outVol, 1-outVol) + EPS) / input:nElement()
   return self.gradInput

end



-- function CustomSoftMaxCriterion:updateOutput(input,target, primitiveSize, worldSize, constrainWeight)
--    -- - log(input) * target - log(1 - input) * (1 - target)
--    local outVol = input[1]
--    local trans = input[2]
--    local scale = input[3]
--    assert( outVol:nElement() == target:nElement(),
--    "input and target size mismatch")
--    local input = input
--    local target = target
--    local EPS = 1 * 10^(-12)
--    local pixel_loss = - (torch.cmul(torch.log(outVol + EPS),target) + torch.cmul(torch.log(1 - outVol + EPS),(1 - target))) 
--    local sum = torch.sum(pixel_loss)
--    local avg_loss = sum / input:nElement()

--    -- add constrain on translate
--    local constrain = primitiveSize * torch.cdiv(1, scale) - torch.cmul(trans-1, scale) - worldSize
--    local constrianLoss =  torch.exp(constrain) - 1
--    local sumConstrainLoss = torch.sum(constrianLoss)

--    local contrainedLoss = avg_loss + constrainWeight * sumConstrainLoss
--    self. weightedConstrainForward = constrainWeight * sumConstrainLoss -- for backward
--    self.output = constrainedLoss
--    return self.output
-- end


-- function CustomSoftMaxCriterion:updateGradInput(input, target,primitiveSize, worldSize)
--    -- - (target - input) / ( input (1 - input) )
--    outVol = input[1]
--    trans = input[2]
--    scale = input[3]   
--    local target = target
--    local gradient = {}
--    assert( outVol:nElement() == target:nElement(),
--    "input and target size mismatch")
--    --backward wrt voxel
--    gradient[1] = torch.cdiv((outVol - target), torch.cmul(outVol, 1-outVol)) / input:nElement()
--    --backward wrt translate
--    gradient[2] = self.weightedConstrainForward * scale * (-1)
--    --backward wrt scale
--    gradient[3] = self.weightedConstrainForward * (- primitiveSize * torch.cdiv(1, torch.pow(scale,2)) - (trans - 1))

--    self.gradInput = gradient
--    return self.gradInput

-- end