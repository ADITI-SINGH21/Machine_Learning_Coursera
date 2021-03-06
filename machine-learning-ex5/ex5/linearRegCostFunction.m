function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

h_x = X*theta;
J =  1/(2*m) * (h_x - y)' * (h_x - y) + (lambda/(2*m)) * (theta(2:length(theta)))' * theta(2:length(theta));
theta_0 = theta;
theta_0(1) = 0;
grad = ((1 / m) * (h_x - y)' * X) + (lambda / m )* theta_0';

grad = grad(:);

end
