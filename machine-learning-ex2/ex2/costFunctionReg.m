function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
J = (1 / m) * ((-y' * log(sigmoid((X*theta)))) - (1 - y)' * log(1 - sigmoid((X*theta)))) + (lambda/(2*m))*(theta(2:length(theta)))' * theta(2:length(theta));
theta_rep = theta;
theta_rep(1) = 0;
grad = (1 / m) * (sigmoid((X*theta)) - y)' * X + (lambda/m)*theta_rep';

end
