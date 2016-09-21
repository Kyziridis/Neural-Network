function h = h(a) % sunartisi energopoihshs..

% h = log(1 + exp(a))

% mathimatiki eustatheia 
m = max(0,a);
h = m +  log( exp(-m) + exp(a-m));

