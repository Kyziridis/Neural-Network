function gradcheck_softmaxlinear(W1, W2,X,T,lambda) 



[K1,D1] = size(W1);
[K2,D2] = size(W2);


% Compute the analytic gradient for W2 
[E, gradEw2,gradientW1] = costgrad_softmax( X, T,W1 , W2,  lambda);

% Scan all parameters to compute 
% numerical gradient estimates
epsilon = 1e-6; 
numgradEw1 = zeros(K1,D1); 
numgradEw2 = zeros(K2,D2); 
%Gradient_check for w2
for k=1:K2
    for d=1:D2
        Wtmp2 = W2; 
        Wtmp2(k,d) = Wtmp2(k,d) + epsilon; 
        Ewplus2 = costgrad_softmax(X, T, W1, Wtmp2, lambda); 
        
        Wtmp2 = W2; 
        Wtmp2(k,d) = Wtmp2(k,d) - epsilon; 
        Ewminus2 = costgrad_softmax(X, T, W1,Wtmp2, lambda);
        
        numgradEw2(k,d) = (Ewplus2 - Ewminus2)/(2*epsilon);
    end
end

% Display the absolute norm as an indication of how close 
% the numerical gradients are to the analytic gradients
diff2 = abs(gradEw2 - numgradEw2);  
disp(['The maximum abolute norm in the gradcheck_2 is : ' num2str(max(diff2(:))) ]);


%Gradient_check for W1
[E, gradEw2,gradientW1] = costgrad_softmax( X, T,W1 , W2,  lambda);

for k=1:K1
    for d=1:D1
        Wtmp1 = W1; 
        Wtmp1(k,d) = Wtmp1(k,d) + epsilon; 
        Ewplus1 = costgrad_softmax( X, T,Wtmp1 , W2,  lambda);
      
        Wtmp1 = W1; 
        Wtmp1(k,d) = Wtmp1(k,d) - epsilon; 
        Ewminus1 = costgrad_softmax( X, T,Wtmp1 , W2,  lambda);
        
        numgradEw1(k,d) = (Ewplus1 - Ewminus1)/(2*epsilon);
    end
end

% Display the absolute norm as an indication of how close 
% the numerical gradients are to the analytic gradients
diff1 = abs(gradientW1 - numgradEw1);  
disp(['The maximum abolute norm in the gradcheck_1 is : ' num2str(max(diff1(:))) ]);








