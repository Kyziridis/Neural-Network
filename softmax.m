function S = softmax(Y) 


[N K] = size(Y);

M = max(Y, [], 2); 

Y = Y - repmat(M, 1, K); 

Y = exp(Y); 

% h pithanotita swstis katataxis gia to y.

S = Y./repmat(sum(Y,2),1,K); 

