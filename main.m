%data initializations%

A=csvread('iris_2.txt');
fprintf("data to process\n");
A=A(randsample(1:length(A),length(A)),:);
A=A(randsample(1:length(A),length(A)),:);
A=A(randsample(1:length(A),length(A)),:);
A=A(randsample(1:length(A),length(A)),:);

pause;
X_main=A(:,1:4)
Y_main=A(:,5:end)



