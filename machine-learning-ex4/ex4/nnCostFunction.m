function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);% m is number of instances.
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%       
%regularized cost function
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% y has size 5000 x 1
K = num_labels;

%Y = eye(K)(y,:); % [5000, 10]
Y = zeros(m, num_labels);
for i = 1 : m
    Y(i, y(i)) = 1;
end

% Part 1
a1 = [ones(m, 1), X]; % results in [5000, 401]
a2 = sigmoid(Theta1 * a1'); % results in [25, 5000]
a2 = [ones(1, size(a2, 2)); a2]; % results in [26, 5000]
h = sigmoid(Theta2 * a2); % results in [10, 5000]

costPositive = -Y .* log(h)';
costNegative =  (1 - Y) .* log(1 - h)';
cost = costPositive - costNegative;

J = (1/m) * sum(cost(:));

% Part 1.4 regularization
Theta1Filtered = Theta1(:,2:end);
Theta2Filtered = Theta2(:,2:end);
reg = (lambda / (2*m)) * (sum(sum(Theta1Filtered.^2)) + sum(sum(Theta2Filtered.^2)));
J = J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

Delta1 = 0;
Delta2 = 0;
for t = 1:m
	% step 1
	a1 = [1; X(t,:)'];
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	% step 2
	yt = Y(t,:)';

	d3 = a3 - yt;

	% step 3
	d2 = (Theta2Filtered' * d3) .* sigmoidGradient(z2);
	%d2 = d2(2:end);


	% step 4
	Delta2 = Delta2 + (d3 * a2');
	Delta1 = Delta1 + (d2 * a1');
end

%step 5
% Delta1 = [25, 401]
% Delta2 = [10, 26]
% Theta1_grad = [25, 401]
% Theta2_grad = [10, 26]
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1Filtered);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2Filtered);

% bias = ones(m, 1);
% X = [bias, X];% add bias unit to the data
% 
% % change y into a matrix
% label_matrix = zeros(m, num_labels);
% for i = 1:m
%     label_matrix(i, y(i)) = 1;    
% end
% 
% % Compute cost
% 
% for i = 1 : m
%     a1 = X(i, :);
%     z2 = a1 * Theta1';% 1* 401 * (25 * 401)T = 1* 25 
%     a2 = sigmoid(z2); % 1* 25
%     a2 = [1 a2];% 1*26
%     z3 = a2 * Theta2'; % 1* 10
%     htheta = sigmoid(z3);% a3
%     delta3 = htheta - label_matrix(i);% 1 * 10
%     gz2_prime = sigmoidGradient(z2);
%     delta2 = delta3 * Theta2(:, 2: end).* gz2_prime;% 1 * 25
% %     temp = delta3' * a2( 2: end);% 10 * 25 temp = [zeros(size(temp, 1) ,
% %     1) temp];
% 
%     Theta1_grad = Theta1_grad + delta2' * a1;
%     Theta2_grad = Theta2_grad + delta3' * a2;
% 
% 
%     
%     %gz1_prime = sigmoidGradient() delta1 = delta2 * Theta1(:, 2: end) .*
%     
%     % compute J
%     for k = 1 : num_labels
%         J = J -label_matrix(i, k) * log(htheta(k)) - (1 - label_matrix(i, k)) ...
%             * log(1- htheta(k));
%     end
% end
% %regularization component
% penalty = 0;
% for j = 1 : hidden_layer_size
%     for k = 2 : size(X,2)
%         penalty = penalty + Theta1(j, k)^2;
%     end
% end
% for j = 1 : num_labels
%     for k = 2 : size(Theta2, 2)
%         penalty = penalty + Theta2(j, k)^ 2;
%     end
% end
% penalty = penalty * lambda / (2 * m);
% J = J / m;
% J = J + penalty;
% % Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1),
% % 1) Theta1(:,2:end)]; Theta2_grad = (1/m) * Theta2_grad + (lambda/m) *
% % [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
% Theta1_grad = (1/m) * Theta1_grad;
% Theta2_grad = (1/m) * Theta2_grad;
% 
% Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda / m) * Theta1(:, 2 : end));
% Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda / m) * Theta2(:, 2 : end));

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
