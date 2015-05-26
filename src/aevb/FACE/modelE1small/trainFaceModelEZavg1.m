% train FACE model that
%   1. BLK factor prior
%   2. unstable sampling method
%   3. diagonal posterior

K = 3;
dimZs = [2 4 8]*K;
algInit = 'ptZ';
shape = 'full';
alg = 'adadelta';
nBlocks = 3;
for paramIter1 = 1:2
    for paramIter2 = 5:12
        % load face data which is renormalized
        [dataTr, ~, ~, ~] = loadFaceData();
        
        % set up the architecture of the network model
        dimZ = dimZs(paramIter1);
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
            NN = pretrainEavg_fullZ(NN, NNsetting, dataTr);
        elseif(strcmp(NN.algInit, 'ptH2'))
            % pretraining H2 layer is problematic
            %   do not recomment using this option
            NN = pretrainH1(NN, NNsetting);
            NN = pretrainEavg_fullZ(NN, NNsetting, dataTr);
            NN = pretrainE_fullH2(NN, NNsetting, dataTr);
        else % random initialization
            NN = defaultNNinit(NN, NNsetting);
        end
        
        %% fine-tune the model
        [BLK, BLKsetting]=defaultBLKsetting(K);
        [BLK]=defaultBLKinit(BLK, NNsetting.mbSize, NN.D2);
        
        % record evolution of the prior model parameters
        process.histSigmaSort = cell(1,1);
        process.histSigma = cell(1,1);
        process.histG = cell(1,1);
        process.histC = cell(1,1);
        process.histId = 1;
        histCount = 1;
        NN.ftLoss = zeros(NNsetting.nEpoch,1);
        
        quickStop=0;
        dataY = zeros(NN.D0, NNsetting.mbSize*NNsetting.L); 
        % auxiliary variables for the efficiency of memory 
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
                firstIdx = (batchIdx-1)*NNsetting.mbSize+1;
                lastIdx = batchIdx*NNsetting.mbSize;
                NN = fullFF(NN, NNsetting, dataTr(:,index(firstIdx:lastIdx)));%, dataY);
                NN.ftLoss(epoch) = NN.ftLoss(epoch)+NN.loss;
                if(rem(batchIdx,20)==1)
                    fprintf(2,'epoch %d, minibatch %d, recon loss: %f\n', epoch, batchIdx, NN.ftLoss(epoch)/batchIdx/NNsetting.mbSize/NNsetting.L);
                end
                
                %% sampling BLK model provided hidden representations
                [BLK, ~] = sampleBLKmemory(NN.Z, BLK, BLKsetting);
                if(rem(batchIdx,100)==0 || (epoch==1 && rem(batchIdx,10)==0))
                    process.histC{histCount} = BLK.C;
                    process.histG{histCount} = BLK.G;
                    process.histSigma{histCount} = BLK.Sigma;
                    process.histSigmaSort{histCount} = BLK.SigmaSort;
                    process.epoch(histCount) = epoch;
                    process.batch(histCount) = batchIdx;
                    histCount = histCount+1;
                end
                
                %% backpropagation
                NN = fullBackProp(NN, NNsetting, BLK);
                
                %% update NN parameter
                NN = updateNN(NN, NNsetting);
            end
            
            %% save model
            fprintf('epoch %d, log-likelihood is %f\n', epoch, NN.ftLoss(epoch)/nSamples/NNsetting.L);
            %if(epoch<=1 || rem(epoch,20)==0)
            if(epoch==1 || epoch==20)
                nameNN = ['modelEmemo_dim' num2str(NN.D2) 'epoch' num2str(epoch) 'Ada' num2str(learnParam) 'init' NN.algInit '.mat'];
                saveModelE(nameNN, NN, BLK, process);
            end
        end
    end
end
