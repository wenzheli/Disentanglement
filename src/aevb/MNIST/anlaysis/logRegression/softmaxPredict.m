function [labels,accs]=softmaxPredict(w,xTr,yTr)

[~, dim] = size(xTr);
% convert w to Ws
numC = length(w)/dim;
Ws = reshape(w, [dim, numC]);
numC = numC+1;
Ws = [zeros(dim,1) Ws];

% calculate the probability for each example
prob = xTr*Ws;
[~, labels] = max(prob,[],2);
[~, labeltrue] = max(yTr,[],2);
accs = sum(labels==labeltrue)/length(labels);

