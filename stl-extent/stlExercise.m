
inputSize  = 28 * 28;
numLabels  = 10;
hiddenSize = 204;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = loadMNISTImages('mnist/train-images-idx3-ubyte');
mnistLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');


unlabeledData = mnistData;

trainData = mnistData;
trainLabels = mnistLabels + 1;
testLabels = testLabels + 1;

fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

opttheta = theta; 

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeledData), ...
                              theta, options);






%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%%======================================================================
%% STEP 4: Train the softmax classifier

softmaxModel = struct;  

lambda = 1e-4;
options.maxIter = 100;
softmaxModel = softmaxTrain(hiddenSize, numLabels, lambda, ...
                            trainFeatures, trainLabels, options);

%for angle = (45 : 45 : 360)
%    for i = 1 : m
%        imgA = reshape(testData(:, i), 28, 28);
%        rotateA = imrotate(imgA, angle);
%        testDate(:, i) = reshape(rotateA, 28 * 28, 1);
%    end
%    testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, testData);
%    softmaxModel = softmaxTrain(hiddenSize, numLabels, lambda, trainFeatures, trainLabels, options);
%end
%http://blog.sina.com.cn/s/blog_4a93ccea0100d1wd.html





%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 



[pred] = softmaxPredict(softmaxModel, testFeatures);

m = size(testData, 2);










%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

