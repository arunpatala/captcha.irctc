require 'nn';

local MultiCrossEntropyCriterion, CrossEntropyCriterion = torch.class('nn.MultiCrossEntropyCriterion', 'nn.CrossEntropyCriterion')

function nn.MultiCrossEntropyCriterion:__init()
    CrossEntropyCriterion.__init(self)
end

function nn.MultiCrossEntropyCriterion:updateOutput(input, target)
    local N = input:size(1)
    local k = input:size(2)
    local C = input:size(3)
    CrossEntropyCriterion.updateOutput(self, input:view(N*k,C), target:view(N*k))
    return self.output
end

function nn.MultiCrossEntropyCriterion:updateGradInput(input, target)
    local N = input:size(1)
    local k = input:size(2)
    local C = input:size(3)
    CrossEntropyCriterion.updateGradInput(self, input:view(N*k,C), target:view(N*k))    
    return self.gradInput:view(N,k,C)
end

return nn.MultiCrossEntropyCriterion