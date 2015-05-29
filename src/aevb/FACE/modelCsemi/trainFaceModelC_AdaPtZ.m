% train FACE model that
%   1. block NW prior
%   2. block diagonal posterior with specified block structure

% alg: optimization algorithms
% algS = {
% 'sgd';
% 'adadelta'};

% algInit: initialization method
% algInitS = {
% 'random';
% 'ptH1';
% 'ptZ';
% 'ptH3'};
% details of these initialization algorithms --> 
%   1. random initialization
%   2. ptH1: 
%       pretrain {W1, b1} using RBM
%       initialize {W2, W3, W4, b2, b3, b4} randomly
%       tie {W5, b5} with {W1, b1}
%   3. ptZ:
%       pretrain {W1, b1} using RBM
%       pretrain {W2, W3, W4, b2, b3, b4} using supervised NN
%       tie {W5, b5} with {W1, b1}
%   4. ptH2:
%       pretrain {W1, b1} using RBM
%       pretrain {W2, W3, W4, b2, b3, b4} using supervised NN
%       pretrain {W5, b5} using supervised NN
% 

clear all
dimZs = [2 4 8]; % number of nodes per block

% global settings:
algInit = 'ptZ';
shape = 'block';
alg = 'adadelta';
nBlocks = 3;

% load the renormalized face data
[dataTr, labelTr, ~, ~] = loadFaceData();

expDesign;

% CASE 2: partially supervised learning using "inverse label"
%   e.g. :
% NN.cWeight(1) = 10;
% NN.cWeight(2) = -10;

for paramIter1 = 1:1%1:2
    for paramIter2 = 10:11  % 12 possible learning parameters
        % set up the architecture of the network model
        NN.learnParam = paramIter2;
        NN.dimZ = dimZs(paramIter1);
        NN.nBlocks = nBlocks;
        % the size of hidden Z layer is dimZ*nBlocks

        NN.shape = shape;
        NN.alg = alg;
        NN.algInit = algInit;
        NN.D0 = size(dataTr,1);
        
        NN.data = 'face';
        [NN, NNsetting] = defaultNNsetting(NN);

        %% initialization and pretraining stage
        if(strcmp(NN.algInit, 'ptH1'))
            NN = pretrainH1(NN, NNsetting);
        elseif(strcmp(NN.algInit, 'ptZ'))
            % it seems that pretraining Z is helpful
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainC_blockZ(NN, NNsetting, dataTr, labelTr);
        elseif(strcmp(NN.algInit, 'ptH2'))
            % pretraining H2 layer is problematic
            %   do not recomment using this option
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainC_blockZ(NN, NNsetting, dataTr, labelTr);
            NN = pretrainC_blockH2(NN, NNsetting, dataTr, labelTr);
        else
            % random initialization
            NN = defaultNNinit(NN, NNsetting);
        end
        
        
        %% fine-tune the model
        NN.ftLossRecon = zeros(NNsetting.nEpoch,1);
        NN.ftLossPred = zeros(NNsetting.nEpoch,NN.nClasses);
        NW = defaultNWinit(NN); % RE-INITIALIZE the NW prior
        
        quickStop=0;
        nSamples = size(dataTr,2);
        numBatch = floor(nSamples/NNsetting.mbSize);
        for epoch = 1:NNsetting.nEpoch
            if(quickStop==1)
                break;
            end

            index = randperm(nSamples);
            for batchIdx = 1:numBatch
                quickStop = 1-verifyValues(NN);
                if(quickStop==1)
                    break;
                end

                %% 1. forward propagation
                firstIdx = (batchIdx-1)*NNsetting.mbSize+1;
                lastIdx = batchIdx*NNsetting.mbSize;
                % thus the size of one minibatch is always constant
                for m=1:NN.nClasses
                    yTr{m} = labelTr{m}(:, index(firstIdx:lastIdx)); 
                end
                NN = blockFF(NN, NNsetting, dataTr(:,index(firstIdx:lastIdx)), yTr, NN.supervised(index(firstIdx:lastIdx)));
                NN.ftLossRecon(epoch) = NN.ftLossRecon(epoch)+NN.lossRecon;
                NN.ftLossPred(epoch,:) = NN.ftLossPred(epoch,:)+NN.lossPred;
                if(rem(batchIdx,20)==1)
                    fprintf(2,'epoch %d, minibatch %d, recon & pred loss: %f, [ %s ]\n', ...
                        epoch, batchIdx, ...
                        NN.ftLossRecon(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L,...
                        num2str(NN.ftLossPred(epoch,:)/batchIdx/NNsetting.mbSize/NNsetting.L/NN.ratioLabel));
                end

                %% 2. update the NW prior model provided hidden representations
                NW = updateNW(NN.Mu, NW);

                %% 3. backpropagation: derivative of (ReconErr + KL-divergence)
                NN = blockBackProp(NN, NNsetting, NW);

                %% 4. update NN parameter
                NN = updateNN(NN, NNsetting);
            end
            
            %% save model
            fprintf('epoch %d, log-likelihood is %f, [ %s ]\n', epoch, ...
                NN.ftLossRecon(epoch)/nSamples/NNsetting.L, ...
                num2str(NN.ftLossPred(epoch,:)/nSamples/NNsetting.L/NN.ratioLabel));

            if(epoch==1 || rem(epoch,5)==0)
                if(strcmp(alg, 'sgd'))
                    nameNN = ['modelC_dim' num2str(NN.D2) 'epoch' num2str(epoch) 'SGD' num2str(NN.learnParam) 'init' NN.algInit '.mat'];
                elseif(strcmp(alg, 'adadelta'))
                    nameNN = ['modelC_dim' num2str(NN.D2) 'epoch' num2str(epoch) 'Ada' num2str(NN.learnParam) 'init' NN.algInit '.mat'];
                end
                %NN.prior = NW;
                saveModelC(nameNN, NN);
            end
        end
    end
end
