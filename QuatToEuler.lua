local QuatToEuler, parent = torch.class('nn.QuatToEuler', 'nn.Module')

function QuatToEuler:__init()
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

function QuatToEuler:check(input)
	local quat = input
	assert(quat:size(2) == 4)
end

local function quaternionToEuler(x,y,z,w)
	local ysqr = torch.pow(y,2)
	local eulerAngle = torch.Tensor(x:size(1),3)
	local t0 = 2.0 * (torch.cmul(w,x) + torch.cmul(y,z))
	local t1 = 1.0 - 2.0 * (torch.pow(x,2) + ysqr)
    local X = torch.atan2(t0,t1)
    eulerAngle:select(2,1):copy(X)

    local t2 = 2.0 * (torch.cmul(w,y) - torch.cmul(z,x))
    t2 = torch.clamp(t2, -1,1)
	local Y = torch.asin(t2)

	eulerAngle:select(2,2):copy(Y)

	local t3 = 2.0 * (torch.cmul(w,z) + torch.cmul(x,y))
	local t4 = 1.0 - 2.0 * (ysqr + torch.pow(z,2))
	local Z = torch.atan2(t3, t4)
	eulerAngle:select(2,3):copy(Z)
	return eulerAngle
end


function QuatToEuler:updateOutput(input)
	local _quat = input
	local quat
	if _quat:nDimension()==1 then
		quat = addOuterDim(_quat)
	else
		quat = _quat
	end
	self:check(quat)
	self.output = quaternionToEuler(quat:select(2,1),quat:select(2,2),quat:select(2,3),quat:select(2,4))
	return self.output
end

-- first angle phi
-- second angle theta
-- third angle psi
local function dtan_dy(y,x)
	return torch.cdiv(x, torch.pow(x,2)+torch.pow(y,2))
end

local function dtan_dx(y,x)
	return -torch.cdiv(y,torch.pow(x,2)+torch.pow(y,2))
end

local function dtan_dq(y,x, dy_dq, dx_dq)
	return torch.cmul(dtan_dy(y,x),dy_dq) + torch.cmul(dtan_dx(y,x), dx_dq)
end

local function dsin_dx(x)
	return torch.pow(torch.sqrt(1-torch.pow(x,2)),-1)
end

local function dsin_dq(x, dx_dq)
	return torch.cmul(dsin_dx(x), dx_dq)
end

function QuatToEuler:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(input):zero()

	local q0 = input:select(2,1) 
	local q1 = input:select(2,2)
	local q2 = input:select(2,3)
	local q3 = input:select(2,4) 
	local Hessian = torch.Tensor(self.gradInput:size(1), gradOutput:size(2), self.gradInput:size(2))
	local dphi_dq0, dphi_dq1, dphi_dq2, dphi_dq3
	local dtheta_dq0, dtheta_dq1, dtheta_dq2, dtheta_dq3
	local dpsi_dq0, dpsi_dq1, dpsi_dq2, dpsi_dq3

	local phi_y = 2 * (torch.cmul(q0,q1) + torch.cmul(q2,q3))
	local phi_x = 1 - 2 * (torch.pow(q1,2) + torch.pow(q2,2))
	local theta_x = 2 * (torch.cmul(q0, q2) - torch.cmul(q3, q1))
	local psi_y = 2 * (torch.cmul(q0, q3) + torch.cmul(q1,q2))
	local psi_x = 1 - 2 * (torch.pow(q2,2) + torch.pow(q3,2))

	local dphi_y_dq0 = 2 * q1
	local dphi_x_dq0 = q0 * 0 -- set to zero, shape as q0
	dphi_dq0 = dtan_dq(phi_y, phi_x, dphi_y_dq0, dphi_x_dq0)
	Hessian:select(3,1):select(2,1):copy(dphi_dq0)

	local dphi_y_dq1 = 2 * q0
	local dphi_x_dq1 = (-4) * q1
	dphi_dq1 = dtan_dq(phi_y, phi_x, dphi_y_dq1, dphi_x_dq1)
	Hessian:select(3,2):select(2,1):copy(dphi_dq1)

	local dphi_y_dq2 = 2 * q3
	local dphi_x_dq2 = (-4) * q2
	dphi_dq2 = dtan_dq(phi_y, phi_x, dphi_y_dq2, dphi_x_dq2)
	Hessian:select(3,3):select(2,1):copy(dphi_dq2)

	local dphi_y_dq3 = 2 * q2
	local dphi_x_dq3 = q3 * 0
	dphi_dq3 = dtan_dq(phi_y, phi_x, dphi_y_dq3, dphi_x_dq3)
	Hessian:select(3,4):select(2,1):copy(dphi_dq3)

	local dtheta_x_dq0 = 2 * q2
	dtheta_dq0 = dsin_dq(theta_x, dtheta_x_dq0)
	Hessian:select(3,1):select(2,2):copy(dtheta_dq0)

	local dtehta_x_dq1 = (-2) * q3
	dtheta_dq1 = dsin_dq(theta_x, dtehta_x_dq1)
	Hessian:select(3,2):select(2,2):copy(dtheta_dq1)

	local dtheta_x_dq2 = 2 * q0
	dtheta_dq2 = dsin_dq(theta_x, dtheta_x_dq2)
	Hessian:select(3,3):select(2,2):copy(dtheta_dq2)

	local dtheta_x_dq3 = (-2) * q1
	dtheta_dq3 = dsin_dq(theta_x, dtheta_x_dq3)
	Hessian:select(3,4):select(2,2):copy(dtheta_dq3)

	local dpsi_y_dq0 = 2 * q3
	local dpsi_x_dq0 = q0 * 0
	dpsi_dq0 = dtan_dq(psi_y, psi_x, dpsi_y_dq0, dpsi_x_dq0)
	Hessian:select(3,1):select(2,3):copy(dpsi_dq0)

	local dpsi_y_dq1 = 2 * q2
	local dpsi_x_dq1 = q1 * 0
	dpsi_dq1 = dtan_dq(psi_y, psi_x, dpsi_y_dq1, dpsi_x_dq1)
	Hessian:select(3,2):select(2,3):copy(dpsi_dq1)

	local dpsi_y_dq2 = 2 * q1
	local dpsi_x_dq2 = (-4) * q2
	dpsi_dq2 = dtan_dq(psi_y, psi_x, dpsi_y_dq2, dpsi_x_dq2)
	Hessian:select(3,3):select(2,3):copy(dpsi_dq2)

	local dpsi_y_dq3 = 2 * q0
	local dpsi_x_dq3 = (-4) * q3
	dpsi_dq3 = dtan_dq(psi_y, psi_x, dpsi_y_dq3, dpsi_x_dq3)
	Hessian:select(3,4):select(2,3):copy(dpsi_dq3)
	
	assert(gradOutput:nDimension() == 2)
	print("q0, \n", Hessian:select(3,1))
	print("q1, \n", Hessian:select(3,2))
	print("q2, \n", Hessian:select(3,3))
	print("q3, \n", Hessian:select(3,4))

	local augmentedGradOutput = torch.Tensor(gradOutput:size(1), 1, gradOutput:size(2))
	augmentedGradOutput:select(2,1):copy(gradOutput)

	self.gradInput = torch.squeeze(torch.bmm(augmentedGradOutput, Hessian))
	return self.gradInput

end
