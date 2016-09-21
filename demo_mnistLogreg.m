% DEMO OF MULTI-CLASS CLASSIFICATION USING A LINEAR MODEL IN THE MNIST DATASET

clear all; close all; 

% Load the MNIST dataset and 
% create the appropritate input and output data matrices 
load mnist_all.mat;
% number of classes
K = 10;
T = []; % the values. i.e which class is it?
X = []; % the data (features)
TtestTrue = []; 
Xtest = [];
Ntrain = zeros(1,10);
Ntest = zeros(1,10);
figure; 
hold on; 

% in sort, this loop loads the training and test data and values
for j=1:10
   
    % Xtmp = double(train1)
    s = ['train' num2str(j-1)];Xtmp = eval(s); Xtmp = double(Xtmp);
     % for each J, all samples in the 'trainJ' variable are for the class J
     % i.e. each sample in train1 are of class '1'
       
    Ntrain(j) = size(Xtmp,1);
    Ttmp = zeros(Ntrain(j), K); 
    Ttmp(:,j) = 1;
    % mark as 1 (true) the column corresponding to j
    % i.e for j=3, the 3rd column of Ttmp will be 1 
    % since Ttmp are the trainin results, they are all of class 3
    X = [X; Xtmp]; % add the sample data (features) of Xtmp in X
    T = [T; Ttmp]; % add the sample values of Xtmp in T
    
    % Xtmp = double(test1), s=double(test2), etc.
    s = ['test' num2str(j-1)]; Xtmp = eval(s); Xtmp = double(Xtmp);
    Ntest(j) = size(Xtmp,1);
    Ttmp = zeros(Ntest(j), K); 
    Ttmp(:,j) = 1; 
    Xtest = [Xtest; Xtmp]; 
    TtestTrue = [TtestTrue; Ttmp]; 
    % again we are adding sample data and sample values in Ttest and TtestTrue respectively
   
    
    ind = randperm(size(Xtmp,1));
    for i=1:10
        subplot(10,10,10*(j-1)+i);     
        imagesc(reshape(Xtmp(ind(i),:),28,28)');
        axis off;
        colormap('gray');     
    end
   
   
end

% normalzie the pixels to take values in [0,1]
X = X/255; Xtest = Xtest/255; 

[N D] = size(X);
% N = how many samples in X
% D = how many features in X
%M = 30;% Hidden Layer size
ent = input("This is my first Neural Network..... if you want to continue press 'enter'...... : ");
M=input("Give the number of hidden units you want, otherwise press '0' for default_number_of_hidden_units  : " );

if M == 0
    M = 250;  % default_number_of_hidden_units     
    fprintf("250 hidden units will be used for the Neural_Network \n")

end
% Add 1 as the first for both the training input and test inputs
% i.e add the bias (1) as the first column of X and Xtest
X = [ones(sum(Ntrain),1), X];
Xtest = [ones(sum(Ntest),1), Xtest ]; 



% Regularization parameter lambda 
lambda = 0.1; 

% Maximum number of iterations of the gradient ascend
options(1) = 100; 
% Tolerance 
options(2) = 1e-6; 
% Learning rate ff
options(3) = 0.5/N;   

% Initial Ws for the gradient ascent
W1init = 0.1*randn(M,D+1);
W2init = 0.1*randn(K, M+1);


%%%%%%%GRADCHECK%%%%%%%

%gradcheck gia W2

% Do a gradient check first
% (in a small random subset of the data so that 
% the gradient check will be fast)
W1check = randn(size(W1init)); 
W2check = randn(size(W2init)); 
ch = randperm(N); 
ch = ch(1:20);
iq = input("If you want Gradient_Check press '1' \n ");
    
if iq==1
    disp('Checking________Begins......');
    gradcheck_softmaxlinear(W1check,W2check,X(ch,:),T(ch,:),lambda); 
end


%%%%%%%%%%%%%%%%%%%%%%%%



% Train the model 
[W1, W2] = ml_softmaxTrain(T, X, lambda, W1init, W2init, options); 


% Test the model 
[Ttest, Ytest]  = ml_softmaxTest(W1,W2, Xtest); 

[~, Ttrue] = max(TtestTrue,[],2); 
err = length(find(Ttest~=Ttrue))/10000;
disp(['The error of the method is: ' num2str(err)])


