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
addpath('../analysis');
dimZs = [2 3 4]; % number of nodes per block

% global settings:
algInit = 'random';
shape = 'block';
alg = 'adadelta';
nBlocks = 2;

for paramIter1 = 1:2
    for paramIter2 = 1:12  % 12 possible learning parameters
            
        % load the renormalized face data
        dataTr = loadMNISTData();
        
        % set up the architecture of the network model
        dimZ = dimZs(paramIter1);
        learnParam = paramIter2;
        [NN, NNsetting] = defaultNNsetting2(dataTr(:,1:10), dimZ, learnParam, shape, alg, nBlocks);
        NN.data = 'mnist';

        %% initialization and pretraining stage
        NN.algInit = algInit;
        if(strcmp(NN.algInit, 'ptH1'))
            NN = pretrainH1(NN, NNsetting);
        elseif(strcmp(NN.algInit, 'ptZ'))
            % it seems that pretraining Z is helpful
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainC_blockZ(NN, NNsetting, dataTr);
        elseif(strcmp(NN.algInit, 'ptH2'))
            % pretraining H2 layer is problematic
            %   do not recomment using this option
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainC_blockZ(NN, NNsetting, dataTr);
            NN = pretrainC_blockH2(NN, NNsetting, dataTr);
        else
            % random initialization
            NN = defaultNNinit(NN, NNsetting);
        end
        
        
        %% fine-tune the model
        NN.ftLoss = zeros(NNsetting.nEpoch,1);
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
                NN = blockFF(NN, NNsetting, dataTr(:,index(firstIdx:lastIdx)));
                NN.ftLoss(epoch) = NN.ftLoss(epoch)+NN.loss;
                if(rem(batchIdx,20)==1)
                    fprintf(2,'epoch %d, minibatch %d, recon loss: %f\n', ...
                        epoch, batchIdx, NN.ftLoss(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L);
                end

                %% 2. update the NW prior model provided hidden representations
                NW = updateNW(NN.Mu, NW);

                %% 3. backpropagation: derivative of (ReconErr + KL-divergence)
                NN = blockBackProp(NN, NNsetting, NW);

                %% 4. update NN parameter
                NN = updateNN(NN, NNsetting);
            end
            
            %% save model
            fprintf('epoch %d, log-likelihood is %f\n', epoch, NN.ftLoss(epoch)/nSamples/NNsetting.L);
            
            if(epoch==1 || epoch==20)
                if(strcmp(alg, 'sgd'))
                    nameNN = ['modelC_dim' num2str(NN.D2) '_' num2str(NN.D1) 'epoch' num2str(epoch) 'SGD' num2str(learnParam) 'init' algInit '.mat'];
                elseif(strcmp(alg, 'adadelta'))
                    nameNN = ['modelC_dim' num2str(NN.D2) '_' num2str(NN.D1) 'epoch' num2str(epoch) 'Ada' num2str(learnParam) 'init' algInit '.mat'];
                end
                saveModelC(nameNN, NN);
            end

        end
    end
end

