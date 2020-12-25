function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum_cost = 0;
sumgrad = zeros(length(theta), 1);
sum_theta = 0;
for i = 1:m
  hypo = sigmoid(X(i, :) * theta);
  sum_cost += (-y(i)) * log(hypo) - (1-y(i)) * log(1 - hypo);
  for i1 = 1:length(theta)    
    sumgrad(i1) += (hypo - y(i)) * X(i, i1);
  endfor
endfor
for i1 = 1:length(theta)  
  if i1 == 1
    grad(i1) = sumgrad(i1) / m;
  else
    sum_theta += theta(i1) ^2;
    grad(i1) = (sumgrad(i1) + theta(i1) * lambda)/ m;
  endif  
endfor
J = sum_cost/m + lambda * sum_theta/(2 * m);





% =============================================================

end
