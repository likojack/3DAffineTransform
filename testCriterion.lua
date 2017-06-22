require 'nn'
require 'stntd'
a = nn.CustomSoftMaxCriterion()
input = torch.Tensor(16,32,32,32,1):fill(0.1)
target = torch.Tensor(16,32,32,32,1):fill(0)
loss = a:forward(input,target)
print(loss)
