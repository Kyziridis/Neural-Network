function [Ttest, Ytest]  = ml_softmaxTest(W1,W2, Xtest)  

% Inputs: 
%         W: the K x (D+1) dimensional matrix of the parameters   
%         Xtest: Ntest x (D+1) input test data with ones already added in the first column 
% Outputs: 
%         Test:  Ntest x 1 vector of the predicted class labels
%         Ytest: Ntest x K matrix of the sigmoid probabilities     
%
% Giorgos Kyziridis & Ioannis Chios (2016)

% Mean predictions

 % Ytest = softmax(Xtest*W2');  % edw isws kati na alla3ei...


N = size(Xtest, 1);

% You need to return the following variables correctly 


h1 = h( Xtest* W1');
Ytest= softmax([ones(N, 1) h1] * W2');


% Hard classification decisions 
[~,Ttest] = max(Ytest,[],2);