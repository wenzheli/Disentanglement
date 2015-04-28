data1 = load('../data/dataOut1a.mat');
data2 = load('../data/dataOutTest1a.mat');

% the following paragraph is used for debugging
% load ../../../data/batchtestdata.mat  
% load ../../../data/batchtraindata.mat
% 
% data1.Z = batchdata';
% data2.Z = batchtestdata';
% data1.L = batchlabel;
% data2.L = batchtestlabel;

dataTr = data1.Z(:,1:50000);
dataVal = data1.Z(:,50001:60000);
dataTe = data2.Z;

labelTr = data1.L(1:50000);
labelVal = data1.L(50001:60000);
labelTe = data2.L;

nSamples = length(labelTr);
nSamplesVal = length(labelVal);
nSamplesTe = length(labelTe);

labelTr = labelTr-min(labelTr)+1;
labelVal = labelVal-min(labelVal)+1;
labelTe = labelTe-min(labelTe)+1;

nClass = max(labelTr);
dimX = size(dataTr,1);
W = rand(dimX, nClass)-0.5;     bestW = W;
bias = rand(nClass, 1);         bestBias = bias;

dW = W*0;
db = bias*0;
momentum = 0.9;
lrate = 0.01;

mbSize = 50;
nEpoch = 50;
bestAcc = 0;
bestAccTe = 0;
for epoch = 1:nEpoch
    index = randperm(nSamples);
    for firstIdx = 1:mbSize:nSamples
        lastIdx = min(firstIdx+mbSize-1, nSamples);
        batchSize = lastIdx-firstIdx+1;
        
        X = dataTr(:,index(firstIdx:lastIdx));
        l = labelTr(index(firstIdx:lastIdx));
        Y = full(sparse(l, 1:batchSize, 1, nClass, batchSize));
        % prediction
        P = bsxfun(@plus, W'*X, bias);
        P = bsxfun(@minus, P, max(P));
        P = exp(P);
        P = bsxfun(@rdivide, P, sum(P));
        
        dW = lrate*X*(Y-P)'/batchSize + momentum*dW;
        db = lrate*sum(Y-P,2)/batchSize + momentum*db;
        
        W = W+dW;
        bias = bias+db;
    end
    
    acc = 0;
    for firstIdx = 1:mbSize:nSamplesVal
        lastIdx = min(firstIdx+mbSize-1, nSamplesVal);
        batchSize = lastIdx-firstIdx+1;
        
        X = dataVal(:,firstIdx:lastIdx);
        l = labelVal(firstIdx:lastIdx);
         
        P = 1./(1+exp(-bsxfun(@plus, W'*X, bias)));
        [~, y] = max(P);
        acc = acc+sum(l(:)==y(:));
    end
    
    
    accTe = 0;
    for firstIdx = 1:mbSize:nSamplesTe
        lastIdx = min(firstIdx+mbSize-1, nSamplesTe);
        batchSize = lastIdx-firstIdx+1;
        
        X = dataTe(:,firstIdx:lastIdx);
        l = labelTe(firstIdx:lastIdx);
         
        P = 1./(1+exp(-bsxfun(@plus, W'*X, bias)));
        [~, y] = max(P);
        accTe = accTe + sum(l(:)==y(:));
    end
    
    
    fprintf(2, 'epoch %d, heldout accuracy %f\n', epoch, acc/nSamplesVal);
    if(acc>bestAcc)
        bestW = W;
        bestBias = bias;
        bestAccTe = accTe/nSamplesTe;
    end
end