local ATP, parent = torch.class('nn.AffineTransformPoint', 'nn.Module')



function ATP:__init()
   parent.__init(self)
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

function ATP:updateOutput(input)
   local _pointList = input[1]
   local transformMatrix = input[2]
   local nPoints = _pointList:size()[1]
   local pointList
   if(_pointList:size()[2] == 3) then
      local homo = torch.Tensor(nPoints,1):fill(1)
      pointList = torch.cat(_pointList, homo, 2)
   else
      pointList = _pointList
   end

   assert(transformMatrix:nDimension()==2
          and transformMatrix:size(1)==3
          and transformMatrix:size(2)==4
          , 'please input affine transform matrices (3x4)')

   self.output:resize(nPoints, 3)
   local output = torch.mm(pointList, transformMatrix:transpose(2,1))
   self.output = output
   return self.output
end

function ATP:updateGradInput(input, gradOutput)
   local _pointList = input[1]
   local transformMatrix = input[2]
   local nPoints = _pointList:size()[1]
   local pointList
   self.gradInput = {}
   if(_pointList:size()[2] == 3) then
      local homo = torch.Tensor(nPoints,1):fill(1)
      pointList = torch.cat(_pointList, homo, 2)
   else
      pointList = _pointList
   end

   assert(transformMatrix:nDimension()==2
          and transformMatrix:size(1)==3
          and transformMatrix:size(2)==4
          , 'please input affine transform matrices (3x4)')

   truncMatrix = transformMatrix:transpose(2,1)[{{1,3},{1,3}}]
   gradPoint = torch.mm(gradOutput, truncMatrix)
   self.gradInput[1] = gradPoint

   self.gradInput[2] = torch.mm(pointList:transpose(2,1), gradOutput):transpose(2,1)

   return self.gradInput
end