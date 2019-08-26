function [J grad] = nnCost(nn_params, input_layer_size,hidden_layer_size,num_labels,X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%Feed Forward%

a1=[ones(m,1),X];
z2=(a1*Theta1');
a2=[ones(size(z2),1),sigmoid(z2)];
h_theta = sigmoid(a2*Theta2');
a3=h_theta;


J = (-sum(sum(y.*log(h_theta))) - sum(sum((1-y).*(log(1-h_theta)))))/m;

%BackPropogation%

s_delta3=(a3-y);
s_delta2 = (s_delta3*Theta2).*[ones(size(z2,1),1) sigmoidGradient(z2)];     % has same dimensions as a2
l_delta1=s_delta2(:,2:end)'* a1;
l_delta2=s_delta3'*a2;
Theta1_grad = Theta1_grad + (1/m) * l_delta1;
Theta2_grad = Theta2_grad + (1/m) * l_delta2;

%regularization%
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));


J = J +  (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) + sum(sum(Theta2(:,2:end).^2)));

grad = [Theta1_grad(:) ; Theta2_grad(:)];


