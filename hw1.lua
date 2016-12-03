-- loading the data
local mnist = require 'mnist';

-- deviding the data to test and train
local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--normalizing our data
local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);


----- ### Shuffling data
function shuffle(data, labels) --shuffle data function
    local randomIndexes = torch.randperm(data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

------   ### Define model and criterion
require 'nn'
require 'cunn'

local inputSize = 28*28
local outputSize = 10 --number of classes, since there are 10 digits
local layerSize = {inputSize, 64,128,256}

-- Creating the model; adding a linear transformation and a transfer function between the layers 
model = nn.Sequential()
model:add(nn.View(28 * 28)) --reshapes the image into a vector without copy
for i=1, #layerSize-1 do
    model:add(nn.Linear(layerSize[i], layerSize[i+1]))
    model:add(nn.ReLU())
end

-- Adding the final layer to the model
model:add(nn.Linear(layerSize[#layerSize], outputSize))
model:add(nn.LogSoftMax())   -- f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)


model:cuda() --ship to gpu
print(tostring(model)) --printing the model

local w, dE_dw = model:getParameters() --w will hold all model parameters and dE_dw will hold the gradient of the loss w.r.t. the same parameters
print('Number of parameters:', w:nElement()) --over-specified model


---- ### Classification criterion
criterion = nn.ClassNLLCriterion():cuda()

---	 ### predefined constants
require 'optim'
batchSize = 128

optimState = {
    learningRate = 0.1
}


--- ### Main evaluation + training function
function forwardNet(data, labels, train)
	timer = torch.Timer()

    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training() --This sets the mode of the Module (or sub-modules) to train=true
    end
    for i = 1, data:size(1) - batchSize, batchSize do --from i=1, while i<=number of rows in data - batchSize, every iteration i increments in batchSize
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda() --the 1st dimention is narrowed from (i) to (i+batchSize-1) -> creating a tensor for current batch
        local yt = labels:narrow(1, i, batchSize):cuda() -- same just for lables
        local y = model:forward(x) --computes the corresponding output of the batch ("y hat")
        local err = criterion:forward(y, yt) --Given y as an in put and yt as a target, computes the loss function assiciated to the criterions and return the result.
        lossAcc = lossAcc + err -- accumulates the losses
        confusion:batchAdd(y,yt) --updates confusion matrix
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
                return err, dE_dw
            end
        
            optim.sgd(feval, w, optimState) --preforms stochastic gradient descent
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches --Calculating avrage loss
    local avgError = 1 - confusion.totalValid
	print(timer:time().real .. ' seconds')

    return avgLoss, avgError, tostring(confusion)
end



--- ### Train the network on training set, evaluate on separate set
epochs = 20

trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end


---		### Introduce momentum, L2 regularization
--reset net weights
model:apply(function(l) l:reset() end)
optimState = {
    learningRate = 0.1,
    momentum = 0.9,
    weightDecay = 1e-3
}

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
end
print('Training error: ' .. trainError[epochs], 'Training Loss: ' .. trainLoss[epochs])
print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs])


--- ### Insert a Dropout layer
model:insert(nn.Dropout(0.9):cuda(), 8) --at each training stage, individual nodes are either dropped out with probablity of 0.1 or kept with propability 0.9

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
end
print('Training error: ' .. trainError[epochs], 'Training Loss: ' .. trainLoss[epochs])
print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs])

-- ********************* Plots *********************
--[[
require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('test.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()
]]