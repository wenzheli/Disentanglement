function [loss,gradient]=softmax(w,xTr,yTr,lambda)
% 1. lambda, the l2-norm coefficient
% 2. belief, used to tell the labeld and unlabeld examples
% note that the last element of w is the bias paramter
% and it has no contribution to the penalty

[N, dim] = size(xTr);
% convert w to Ws
numC = length(w)/dim;
Ws = reshape(w, [dim, numC]);
numC = numC+1;
Ws = [zeros(dim,1) Ws];

% calculate the probability for each example
prob = xTr*Ws;
prob = bsxfun(@minus, prob, max(prob,[],2));
prob = exp(prob);
prob = bsxfun(@rdivide, prob, sum(prob,2));

% calculate the loss
probs = prob(:);
labels = yTr(:);
loss = -sum(log(probs(labels>0)))/N +lambda*(w'*w); % w is col vector
% L = 0;
% for n=1:N
%     L = L - log(prob(n,yTr(n,:)>0));
% end
% 
% keyboard

% calculate the gradients
% gradient = Ws*0; % shape: dim * numC
delta = yTr-prob;
gradient = -xTr' * delta;

% gradient2 = Ws*0;
% for l=1:numC
%     for n = 1:N
%         gradient2(:,l) = gradient2(:,l) - (yTr(n,l)-prob(n,l))*transpose(xTr(n,:));
%     end
% end

% transform the gradients
gradient(:,2:end) = bsxfun(@minus, gradient(:,2:end), gradient(:,1));
gradient(:,1) = [];
gradient = gradient(:)/N+lambda*w;

