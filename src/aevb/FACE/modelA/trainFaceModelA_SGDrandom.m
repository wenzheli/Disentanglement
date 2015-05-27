dimZs = [2 3];
nBlocks = 3;
% nBlocks*dimZ: the number of hidden nodes
algInit = 'random';
shape = 'diag';
alg = 'sgd';
        
for paramIter1 = 1:2
    for paramIter2 = 1:6 % if SGD, totally 4 parameters
        % load face data which is renormalized
        [dataTr, ~, ~, ~] = loadFaceData();
        
        % set up the architecture of the network model
        dimZ = dimZs(paramIter1)*nBlocks;
        learnParam = paramIter2;
        
        [NN, NNsetting] = defaultNNsetting(dataTr(:,1:10), dimZ, learnParam, shape, alg, nBlocks);
        NN.data = 'face';
        
        nSamples = size(dataTr,2);
        numBatch = floor(nSamples/NNsetting.mbSize);
        
        %% initialization and pretraining stage
        NN.algInit = algInit;
        if(strcmp(NN.algInit, 'ptH1'))
            NN = pretrainH1(NN, NNsetting);
        elseif(strcmp(NN.algInit, 'ptZ'))
            % it seems that pretraining Z is helpful
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainA_diagZ(NN, NNsetting, dataTr);
        elseif(strcmp(NN.algInit, 'ptH2'))
            % pretraining H2 layer is problematic
            %   do not recomment using this option
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainA_diagZ(NN, NNsetting, dataTr);
            NN = pretrainA_diagH2(NN, NNsetting, dataTr);
        else % random initialization
            NN = defaultNNinit(NN, NNsetting);
        end
        
        %% fine-tune the model
        % record evolution of the prior model parameters
        histCount = 1;
        NN.ftLoss = zeros(NNsetting.nEpoch,1);
        
        quickStop=0;
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
                %% forward propagation
                %   call <diagFF> or <blockFF>
                firstIdx = (batchIdx-1)*NNsetting.mbSize+1;
                lastIdx = batchIdx*NNsetting.mbSize;
                NN = diagFF(NN, NNsetting, dataTr(:,index(firstIdx:lastIdx)));
                NN.ftLoss(epoch) = NN.ftLoss(epoch)+NN.loss;
                if(rem(batchIdx,20)==1)
                    fprintf(2,'epoch %d, minibatch %d, recon loss: %f\n', epoch, batchIdx, NN.ftLoss(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L);
                end

                
                %% backpropagation
                NN = diagBackProp(NN, NNsetting);
                
                %% update NN parameter
                NN = updateNN(NN, NNsetting);
            end
            %% save model
            fprintf('epoch %d, log-likelihood is %f\n', epoch, NN.ftLoss(epoch)/nSamples/NNsetting.L);

            if(epoch==1 || epoch==20)
                if(strcmp(alg, 'sgd'))
                    nameNN = ['modelA_dim' num2str(NN.D2) 'epoch' num2str(epoch) 'SGD' num2str(learnParam) 'init' algInit '.mat'];
                elseif(strcmp(alg, 'adadelta'))
                    nameNN = ['modelA_dim' num2str(NN.D2) 'epoch' num2str(epoch) 'Ada' num2str(learnParam) 'init' algInit '.mat'];
                end
                saveModelA(nameNN, NN);
            end

        end
    end
end
