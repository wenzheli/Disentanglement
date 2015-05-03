% test Gaussian-Binary RBM
load ../data/face.mat
dataTr = single(data(:,1:end-2));
clear data;
dataTr = dataTr/max(max(dataTr)); % normalization
dataTr = dataTr';

% randomly split the training data into 
%   90% training samples
%   10% validation samples
N = size(dataTr,2);
nTr = floor(N*0.9);
nVal = N-nTr;

index = randperm(N);
dataVal = dataTr(:,index(nTr+1:N));
dataTr = dataTr(:,index(1:nTr));

faceM = mean(dataTr,2);
faceSTD = std(dataTr, [], 2);

dataTr = bsxfun(@minus, dataTr, faceM);
dataTr = bsxfun(@rdivide, dataTr, faceSTD);

dataVal = bsxfun(@minus, dataVal, faceM);
dataVal = bsxfun(@rdivide, dataVal, faceSTD);

load faceParams.mat
deviceID = 3;
for id = 1:size(parameters,1)
    fprintf(2,'training using hyperparameter %d\n', id);
    name = ['model1Khu' num2str(id) '_'];
    % pretrain: [momentum, l2, lrate1 lrate2]
    % L1NN: [momentum, l2, lrate, scaling]
    % L2NN: [momentum, l2, lrate, scaling]

    % hidden layer 1 parameters
    params.minBatchSize = 100;
    params.numHiddenUnits = 1000;
    params.preTrain_LearningRate = parameters(id,1);%0.0003;
    params.preTrain_Momentum1 = parameters(id,2);%0.3;
    params.preTrain_Momentum2 = parameters(id,3);%0.3;
    params.numPreTrain_Epochs = 40;
    params.momChange_Epoch = parameters(id,4);
    params.weightcost = parameters(id,5);%0.000002;
    paramsl1 = params

    fprintf(2,'training layer 1\n');
    ptl1name = [name 'ptL1'];
    [W1, visBias1, hidBias1, best_dev_recerr] = gGrbm(deviceID, dataTr, dataVal, paramsl1, ptl1name);
    ptl1matname = [ptl1name 'NN.mat'];
    save(ptl1matname,'W1','visBias1','hidBias1','best_dev_recerr','faceM','faceSTD');
end
