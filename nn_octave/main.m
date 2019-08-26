clear ; close all; clc
pkg load statistics


%data initializations%

A=csvread('iris_2.txt');
fprintf("data to process\n");
A=A(randsample(1:length(A),length(A)),:);
A=A(randsample(1:length(A),length(A)),:);
A=A(randsample(1:length(A),length(A)),:);
A=A(randsample(1:length(A),length(A)),:);

X_main=A(:,1:4);
Y_main=A(:,5:end);
X_main=[X_main,X_main(:,1).*X_main(:,2),X_main(:,2).*X_main(:,3),X_main(:,3).*X_main(:,4),X_main(:,4).*X_main(:,1),X_main(:,1).*X_main(:,2).*X_main(:,3),X_main(:,4).*X_main(:,2).*X_main(:,3),X_main(:,1).*X_main(:,2).*X_main(:,4)];
%%nn parameters
input_layer_size = 11;
hidden_layer_size=16;
lables=3;

%%test data and train data
X=X_main(1:112,:);
Y=Y_main(1:112,:);

m = size(X, 1);
lambda=0;
fprintf('test J')
Theta1=rand(hidden_layer_size,input_layer_size+1);
Theta2=rand(lables,hidden_layer_size+1);
nn_params = [Theta1(:) ; Theta2(:)];
J = nnCost(nn_params,input_layer_size,hidden_layer_size, lables,X, Y, lambda)
pause;

initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size,lables);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

J = nnCost(initial_nn_params,input_layer_size,hidden_layer_size, lables,X, Y, lambda)
pause;
%check if nn is fine%
checkNNGradients;

lambda = 3;
checkNNGradients(lambda);

%NN training
fprintf('\nTraining Neural Network... \n')
iter=5000;

options = optimset('MaxIter', iter);
lambda = 1;


costFunction = @(p) nnCost(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   lables, X, Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 lables, (hidden_layer_size + 1));



%%Cross Validation

X_train=X_main(112:135,:);
Y_train=Y_main(112:135,:);

lambda_history=zeros(1000,1);

for i=1:1000
    lambda_history(i)= nnCost(nn_params,input_layer_size,hidden_layer_size, lables,X_train, Y_train, i/10);
end
[minval,row]=min(min(lambda_history));
lambda=row;
plot(lambda_history,1:1000,'x')
X_test=X_main(136:end,:);
Y_test=Y_main(136:end,:);

error_test=nnCost(nn_params,input_layer_size,hidden_layer_size, lables,X_test, Y_test, lambda)*100/length(X_test)

error_on_full_data=nnCost(nn_params,input_layer_size,hidden_layer_size, lables,X_main, Y_main, lambda)*100/length(X_test)