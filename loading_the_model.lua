function returnAvgError()
	-- loading the data
	local mnist = require 'mnist';
	require 'nn'
	require 'cunn'
	---	 ### predefined constants
	require 'optim'


	-- deviding the data to test and train
	local trainData = mnist.traindataset().data:float();
	local trainLabels = mnist.traindataset().label:add(1);
	testData = mnist.testdataset().data:float();
	testLabels = mnist.testdataset().label:add(1);

	--normalizing our test data w.r.t. the train mean and std
	local mean = trainData:mean()
	local std = trainData:std()
	testData:add(-mean):div(std);

	-- tranfering to cuda
	testData = testData:cuda()
	testLabels = testLabels:cuda()

	--load the model (the trained net)
	model = torch.load('nn_hw1.t7')
	model:evaluate() --turn off drop out

	--calculating the estimated labels with the trained nn
	local y_hat = model:forward(testData)
	-- creating and calculating confusion matrix
	local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	confusion:batchAdd(y_hat,testLabels)

	-- calculating average error
	confusion:updateValids()
	local avgError = 1 - confusion.totalValid

	-- returning the average error
	return avgError
end

print ('avgError is :',returnAvgError())