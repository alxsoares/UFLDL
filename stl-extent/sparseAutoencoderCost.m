function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));


Jcost = 0;
Jweight = 0;
Jsparse = 0;
[n m] = size(data);

%forward
z2 = W1 * data + repmat(b1, 1, m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2, 1, m);
a3 = sigmoid(z3);

%calculate cost
Jcost = (1. / (2. * m)) * sum(sum((a3 - data) .^ 2));
Jweight = (1. / 2) * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));
rho = (1. / m) .* sum(a2, 2);
Jsparse = sum(sparsityParam .* log(sparsityParam ./ rho) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1 - rho)));
cost = Jcost + lambda * Jweight + beta * Jsparse;

%backprop
delta3 = -(data - a3) .* sigmoidGrad(z3);
sterm = beta * (-sparsityParam ./ rho + (1 - sparsityParam) ./ (1 - rho));

delta2 = (W2' * delta3 + repmat(sterm, 1, m)) .* sigmoidGrad(z2);

%calculate Weight Gradient
W2grad = (1. / m) * delta3 * a2' + lambda * W2;
b2grad = (1. / m) * sum(delta3, 2);
W1grad = (1. / m) * delta2 * data' + lambda * W1;
b1grad = (1. / m) * sum(delta2, 2);





%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function sigmgrad = sigmoidGrad(x)

    sigmgrad = sigmoid(x) .* (1 - sigmoid(x));
end


