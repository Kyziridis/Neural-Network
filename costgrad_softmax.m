function [E, gradEw2,gradientW1] = costgrad_softmax( X, T,W1 , W2,  lambda)

K = size(W2,1);



% FeedForward 
    % Layer-2
    Z = h(X*W1'); 
    % Layer- 3
    Z = [ones(size(Z,1),1) Z]; % add the bias of the hidden layer
    Y = Z*W2';  %pinakas S ekfwnisis

M = max(Y, [], 2);
%aferoume bias apo W2


E = sum(sum( T.*Y )) - sum(M)  - sum(log(sum(exp(Y - repmat(M, 1, K)), 2)))  - (0.5*lambda)*sum(sum(W2.*W2)); 


% Return also the gradients if neeeded
if nargout > 1  
   % softmax probabilities 
   S = softmax(Y);
   % gradient_W2
   gradEw2 = ((T - S)')*Z - lambda*W2;      % gia W2
   
   % gradient_W1
    hta = sigmoid(X*W1');
    delta_k = ( T-S)'; % alla3ame apo (Y-T)'
    W2 = W2(: , 2:end);% aferoume bias
    tmp_j =W2' *delta_k;
    delta_j = tmp_j.*hta';
    gradientW1 = delta_j* X;
   
end
