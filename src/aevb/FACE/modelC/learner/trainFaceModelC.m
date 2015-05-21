% train FACE model that
%   1. block NW prior
%   2. block diagonal posterior with specified block structure

% alg: optimization algorithms
% algInit: initialization method

% algS = {
% 'sgd';
% 'adadelta'};

% algInitS = {
% 'random';
% 'ptH1';
% 'ptH2';
% 'ptH3'};

% initialization algorithms
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

dimZs = [2 6 18];
algInit = 'ptH1';
for paramIter1 = 1:1
    for paramIter2 = 11:12
        
        % load face data which is renormalized
        [dataTr, ~, ~, ~] = loadFaceData();
        
        % set up the architecture of the network model
        dimZ = dimZs(paramIter1);
        learnParam = paramIter2;
        shape = 'block';
        alg = 'adadelta';
        nBlocks = 3;
        [NN, NNsetting] = defaultNNsetting(dataTr(:,1:10), dimZ, learnParam, shape, alg, nBlocks);
       
        
        %% initialization and pretraining stage
        NN.algInit = algInit;
        if(strcmp(NN.algInit, 'ptH1'))
            NN = pretrainH1(NN, NNsetting);
        elseif(strcmp(NN.algInit, 'ptZ'))
            % it seems that pretraining Z is helpful
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainBlockZ(NN, NNsetting, dataTr);
        elseif(strcmp(NN.algInit, 'ptH2'))
            % pretraining H2 layer is problematic
            %   do not recomment using this option
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainBlockZ(NN, NNsetting, dataTr);
            NN = pretrainBlockH2(NN, NNsetting, dataTr);
        else % random initialization
            NN = defaultNNinit(NN, NNsetting);
        end
        

        %% fine-tune the model
        NN.ftLoss = zeros(NNsetting.nEpoch,1);
        NW = defaultNWinit(NN);
        quickStop=0;
        nSamples = size(dataTr,2);
        numBatch = floor(nSamples/NNsetting.mbSize);
        for epoch = 1:NNsetting.nEpoch
            if(quickStop==1)
                break;
            end

            index = randperm(nSamples);
            for batchIdx = 1:numBatch
                if(quickStop==1)
                    break;
                end
                %% forward propagation
                %   call <diagFF> or <blockFF>
                firstIdx = (batchIdx-1)*NNsetting.mbSize+1;
                lastIdx = batchIdx*NNsetting.mbSize;
                NN = blockFF(NN, NNsetting, dataTr(:,index(firstIdx:lastIdx)));
                NN.ftLoss(epoch) = NN.ftLoss(epoch)+NN.loss;
                if(rem(batchIdx,20)==1)
                    fprintf(2,'epoch %d, minibatch %d, recon loss: %f\n', ...
                        epoch, batchIdx, NN.ftLoss(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L);
                end

                %% update the NW prior model provided hidden representations
                NW = updateNW(NN.Mu, NW);

                %% backpropagation
                NN = blockBackProp(NN, NNsetting, NW);

                %% update NN parameter
                NN = updateNN(NN, NNsetting);
            end
            %% save model
            fprintf('epoch %d, log-likelihood is %f\n', epoch, NN.ftLoss(epoch)/nSamples/NNsetting.L);
            if(epoch<=3 || rem(epoch,5)==0)
                nameNN = ['modelC_dim' num2str(NN.D2) 'epoch' num2str(epoch) 'lparam' num2str(learnParam) 'init' NN.algInit '.mat'];
                NN.prior = NW;
                saveModel(nameNN, NN);
            end

        end

    end
end
