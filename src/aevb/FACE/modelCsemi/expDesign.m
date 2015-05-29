% design such an experiment
% 1. block 1 use part of view label to train
% 2. block 2 and 3 use the same part of view label to train, inversely


labelTr{2} = labelTr{1};
labelTr{3} = labelTr{1};

if(iscell(labelTr))
    NN.nClasses = length(labelTr);          % No. of groups of labels
    NN.sizeClasses = zeros(NN.nClasses,1);  % the number of classes in each group of label
    
    for classID = 1:NN.nClasses
        labelTr{classID} = double(labelTr{classID} - min(labelTr{classID}) + 1);
        NN.sizeClasses(classID) = max(labelTr{classID});
        labelTr{classID} = sparse(labelTr{classID}, 1:length(labelTr{classID}), 1, max(labelTr{classID}), length(labelTr{classID}));
    end
else
    NN.nClasses = 1;
    labelTr = double(labelTr-min(labelTr)+1);
    NN.sizeClasses = max(labelTr{1});
    labelTr{1} = sparse(labelTr, 1:length(labelTr), 1, max(labelTr), length(labelTr));
end

%% settings relevant to classification task:
% 1. randomly select RATIO of all samples
nSamples = size(dataTr,2);
NN.ratioLabel = 0.1;
labelIdx = randperm(nSamples, floor(NN.ratioLabel*nSamples));
NN.supervised = zeros(1,nSamples);
NN.supervised(labelIdx) = 1; % indicator vector whether labels are used


% 2. create a map between blocks and labels
%   the default setting is: the i-th block correspond to the i-th label
NN.mapBlock = zeros(NN.nClasses,1);
for i=1:NN.nClasses
    NN.mapBlock(i) = i;
end

%% hyperparameters on FACE dataset:
% CASE 1: fully supervised learning
%   cWeight(1): classifier weight on class 1 (view)
%   cWeight(2): classifier weight on class 2 (ID)
%   cWeight(3): classifier weight on class 3 (brightness)
%       e.g. :
NN.cWeight(1) = 100;
NN.cWeight(2) = -100;
NN.cWeight(3) = -100;