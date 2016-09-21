function [W1, W2] = ml_softmaxTrain(T, X, lambda, W1init, W2init, options)

%
% What it does: It trains using backpropagation the Weights of a NN with 
% 1 hidden layer
%
% Inputs: 
%         T: N x K binary output data matrix indicating the classes
%         X: N x (D+1) input data vector with ones already added in the first column
%         lambda: the positive regularizarion parameter
%         W1init: M x (D+1) matrix of the initial values of the parameters
%         W2init: K x (M+1) matrix of the initial values of the parameters  
%         options: options(1) is the maximum number of iterations 
%                  options(2) is the tolerance
%                  options(3) is the learning rate eta 
% Outputs: 
%         W1, W2: the trained weight matrices
%  
% George Kyziridis & Ioannis Chios (2016)

W1 = W1init; W2 = W2init;

%K = size(W2,1); 
%M = size(W1,1); 

% Maximum number of iteration of gradient ascend
iter = options(1); 
% Tolerance
tol = options(2);
% Learning rate
eta = options(3);

%feedfor   
Ewold = -Inf; 
for it=1:iter

    % FeedForward 
    
    % Cost and its gradient for last layer
    [E, gradEw2,gradientW1 ] = costgrad_softmax(X, T,W1, W2, lambda);
    
  % [E1, gradientW1] = cost_w1(W1, W2, Y,T,  X , lambda);

    fprintf('Iteration: %d, Cost: %f\n',it, E);
  

    % Break if you achieve the desired accuracy in the cost function
    if abs(E - Ewold) < tol 
        break;
    end
    
    % Update parameters based on gradient ascent 
    
  
     W1 = W1 + eta*gradientW1;
     W2 = W2 + eta*gradEw2;
    
    % keep the old value of the cost for next iteration
    Ewold = E; 
%
end
