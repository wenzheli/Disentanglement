function [W, visBias, hidBias, best_dev_recerr ] = gGrbm(deviceID, data, devdata, params,name)
% Pre-train Gaussian-Bernoulli RBM with CD
%
%  data:  training data, D x N matrix
%  devdata: dev data, D x M matrix (could be [])
%  params
    
    gpuDevice(deviceID);
    
    randseed = 1;
    
    fp = fopen(name, 'w');
    mbSize = params.minBatchSize;
    numHids = params.numHiddenUnits;
    [numDims, nSamples] = size(data);
    [~, nSamplesDev] = size(devdata);
    lRate = params.preTrain_LearningRate;
    momentum = params.preTrain_Momentum1;
    momChange = params.momChange_Epoch;
    numEpochs = params.numPreTrain_Epochs;
    weightcost = params.weightcost;
    
    fprintf(fp, 'hyperparameters:\n');
    fprintf(fp, 'lrate: %f\n', lRate);
    fprintf(fp, 'momentums: %f, %f, %d\n', momentum, params.preTrain_Momentum2, momChange);
    fprintf(fp, 'l2: %f\n', weightcost);
    % allocate space and initialize values
    initRange = sqrt(6)/sqrt(numDims+numHids);
    W = (rand(numDims, numHids)-0.5)*initRange*2;
    visBias = mean(data,2);%-ones(numDims,1);
    hidBias = zeros(numHids,1);
    
    bestW = W;
    bestVisBias = visBias;
    bestHidBias = hidBias;

    fprintf(fp,'===== Pretraining of Gaussian-Binary RBM ===== \n');
    fprintf(fp,'Total samples: %d, Dim: %d, # of hidden: %d\n', nSamples, numDims, numHids);
    
    % sufficient statistics
    visData     = gpuArray.zeros(numDims, mbSize, 'single');
    negData     = gpuArray.zeros(numDims, mbSize, 'single');
    hidProb     = gpuArray.zeros(numHids, mbSize, 'single');
    hidStates   = gpuArray.zeros(numHids, mbSize, 'single');
    negHidProb  = gpuArray.zeros(numHids, mbSize, 'single');
    
    incW        = gpuArray.zeros(size(W,1), size(W,2), 'single');
    incVisBias  = gpuArray.zeros(size(visBias,1), size(visBias,2), 'single');
    incHidBias  = gpuArray.zeros(size(hidBias,1), size(hidBias,2), 'single');
    
    W           = gpuArray(W);
    visBias     = gpuArray(visBias);
    hidBias     = gpuArray(hidBias);
    
    
    w_posUpdate         = gpuArray.zeros(size(W,1), size(W,2),'single');
    visBias_posUpdate   = gpuArray.zeros(size(visBias,1), size(visBias,2),'single');
    hidBias_posUpdate   = gpuArray.zeros(size(hidBias,1), size(hidBias,2),'single');
    w_negUpdate         = gpuArray.zeros(size(W,1), size(W,2),'single');
    visBias_negUpdate   = gpuArray.zeros(size(visBias,1), size(visBias,2),'single');
    hidBias_negUpdate   = gpuArray.zeros(size(hidBias,1), size(hidBias,2),'single');
    
%     visData     = gpuArray.zeros(numDims, mbSize, 'single');
%     negData     = gpuArray.zeros(numDims, mbSize, 'single');
%     hidProb     = gpuArray.zeros(numHids, mbSize, 'single');
%     hidStates   = gpuArray.zeros(numHids, mbSize, 'single');
%     negHidProb  = gpuArray.zeros(numHids, mbSize, 'single');
%              
%     W          = gpuArray(single(W));
%     visBias    = gpuArray(single(visBias));
%     hidBias    = gpuArray(single(hidBias));
% 
%     incW        = gpuArray.zeros(size(W,1), size(W,2), 'single');
%     incVisBias  = gpuArray.zeros(size(visBias,1), size(visBias,2), 'single');
%     incHidBias  = gpuArray.zeros(size(hidBias,1), size(hidBias,2), 'single');
    %% epoch 0: before learning, 
    %  estimate the reconstruction error from random initialization
    
    epoch0=1;
    nSubSamples = floor(nSamples/10);
    if(epoch0==1)
        recerr = 0;
        index = randperm(nSamples);
        for firstInd = 1:mbSize:nSubSamples
            lastInd = min(firstInd+mbSize-1, nSamples);
            batchSize = lastInd-firstInd+1;
            visData(:,1:batchSize) = data(:, index(firstInd:lastInd));
            hidProb(:,1:batchSize) = 1./ (1+exp(-(bsxfun(@plus, W'*visData(:,1:batchSize), hidBias))));
            % do we need sampling during reconstruction error estimation?
            hidStates(:,1:batchSize) = double(hidProb(:,1:batchSize) > rand(numHids,batchSize,'single'));
            negData(:,1:batchSize) = bsxfun(@plus, W*hidStates(:,1:batchSize),visBias);
            
            recerr =  sum(sum((visData(:,1:batchSize) - negData(:,1:batchSize)).^2))+recerr;
        end
        
        if ~isempty(devdata)
            dev_recerr=0;
            for firstInd = 1:mbSize:nSamplesDev
                lastInd = min(firstInd+mbSize-1, nSamplesDev);
                batchSize = lastInd-firstInd+1;
                visData(:,1:batchSize) = devdata(:, firstInd:lastInd);
                hidProb(:,1:batchSize) = 1./ (1+exp(-(bsxfun(@plus, W'*visData(:,1:batchSize), hidBias))));
                % do we need sampling during reconstruction error estimation?
                hidStates(:,1:batchSize) = double(hidProb(:,1:batchSize) > rand(numHids,batchSize,'single'));
                negData(:,1:batchSize) = bsxfun(@plus, W*hidStates(:,1:batchSize),visBias);
                dev_recerr =  sum(sum((visData(:,1:batchSize) - negData(:,1:batchSize)).^2))+dev_recerr;
            end
            dev_recerr = dev_recerr/nSamplesDev;
            
            fprintf(fp,'ep: 0, error: %e, dev_error: %e\n ',  recerr/nSubSamples,dev_recerr);
            fprintf('ep: 0, error: %e, dev_error: %e\n ',  recerr/nSubSamples,dev_recerr);
        else
            fprintf(fp,'ep: 0, error: %e\n ', recerr/nSamples);
            fprintf('ep: 0, error: %e\n ', recerr/nSamples);
        end
    end

    if(~isempty(devdata))
        best_dev_recerr = gather(dev_recerr);
    else
        best_dev_recerr = 1e10;
    end

    boolNan =0; % if the reconstruction error is NAN, break
    %recerrs = 0;
    for epoch = 1:numEpochs
        batchId = 0;
        if(epoch>=momChange)
            momentum = params.preTrain_Momentum2;
        end
        
        startTime = tic;
        recerr = 0;
        if(boolNan==1)
            break;
        end
        
        index = randperm(nSamples);
        for firstInd = 1:mbSize:nSamples
            batchId = batchId+1;
            lastInd = min(firstInd+mbSize-1, nSamples);
            batchSize = lastInd-firstInd+1;
            visData(:,1:batchSize) = data(:, index(firstInd:lastInd));
            
            if(isnan(gather(sum(sum(W)))))
                boolNan = 1;
                %keyboard
                break;
            end
            
            % ---> oneStepGibbs
            % 1. propagage forward to get hidden
            hidProb(:,1:batchSize) = ...
                1./ (1+exp(-(bsxfun(@plus, W'*visData(:,1:batchSize), hidBias))));
            hidStates(:,1:batchSize) = ...
                double(hidProb(:,1:batchSize) > rand(numHids,batchSize,'single'));

            % 2. from hidden to visible -- to generate Gaussian visible unit, we only need the
            % mean, which is the marginalized Gaussian unit
            negData(:,1:batchSize) = ...
                bsxfun(@plus, W*hidStates(:,1:batchSize),visBias);
            % 3. forward again
            negHidProb(:,1:batchSize) = ...
                1./ (1+exp(-(bsxfun(@plus, W'*negData(:,1:batchSize), hidBias))));  
            % <--- oneStepGibbs
            
            % ---> CD_Update
            % positive statistics
            w_posUpdate = visData(:,1:batchSize)*hidProb(:,1:batchSize)';
            visBias_posUpdate = sum(visData(:,1:batchSize),2);
            hidBias_posUpdate = sum(hidProb(:,1:batchSize), 2);
            % negative statistics
            w_negUpdate = negData(:,1:batchSize)*negHidProb(:,1:batchSize)';
            visBias_negUpdate = sum(negData(:,1:batchSize),2);
            hidBias_negUpdate = sum(negHidProb(:,1:batchSize),2);

            % Update: momentum
            incW = incW*momentum;
            incVisBias = incVisBias*momentum;
            incHidBias = incHidBias*momentum;

            % Update: adding statistics
            incW = incW + ((w_posUpdate - w_negUpdate)/batchSize- weightcost*W)*lRate;
            incVisBias = incVisBias + (visBias_posUpdate - visBias_negUpdate)*lRate/batchSize;   
            incHidBias = incHidBias + (hidBias_posUpdate - hidBias_negUpdate)*lRate/batchSize;
            % Update:
            W = W + incW;
            visBias = visBias + incVisBias;
            hidBias = hidBias + incHidBias;
            % <--- CD_Update
            
            recerr = recerr + sum(sum( (visData(:,1:batchSize) - negData(:,1:batchSize)).^2));
            %recerrs = [recerrs; sum(sum( (visData(:,1:batchSize) - negData(:,1:batchSize)).^2))];
            
       
            %W = gather(W);
            %visBias = gather(visBias);
            %hidBias = gather(hidBias);
            %dW = gather(incW);
            %dVisBias = gather(incVisBias);
            %dHidBias = gather(incHidBias);	
            %matname = [name '_' num2str(epoch) '.mat'];
            %save(matname, 'W','hidBias','visBias','params','dW','dVisBias','dHidBias');
            
            %if(rem(batchId,10)==0 || lastInd == nSamples)
            if(rem(batchId,100)==0 || lastInd == nSamples)
                % calculate reconstruction error on heldout data
                if ~isempty(devdata)
                    dev_recerr=0;
                    for firstDevInd = 1:mbSize:nSamplesDev
                        lastDevInd = min(firstDevInd+mbSize-1, nSamplesDev);
                        batchSize = lastDevInd-firstDevInd+1;
                        visData(:,1:batchSize) = devdata(:, firstDevInd:lastDevInd);
                        hidProb(:,1:batchSize) = 1./ (1+exp(-(bsxfun(@plus, W'*visData(:,1:batchSize), hidBias))));
                        % do we need sampling during reconstruction error estimation?
                        hidStates(:,1:batchSize) = double(hidProb(:,1:batchSize) > rand(numHids,batchSize,'single'));
                        negData(:,1:batchSize) = bsxfun(@plus, W*hidStates(:,1:batchSize),visBias);
    
                        dev_recerr =  sum(sum((visData(:,1:batchSize) - negData(:,1:batchSize)).^2))+dev_recerr;
                    end

                    dev_recerr=dev_recerr/nSamplesDev;
                    if(best_dev_recerr>dev_recerr)
                        best_dev_recerr = gather(dev_recerr);
                        bestW = W;
                        bestHidBias = hidBias;
                        bestVisBias = visBias;
                    end

                    fprintf(fp,'ep: %d, batch %d, error: %e, dev_error: %e\n', epoch, batchId, recerr/(batchId*mbSize),dev_recerr);
                    fprintf('ep: %d, batch %d, error: %e, dev_error: %e\n', epoch, batchId, recerr/(batchId*mbSize),dev_recerr);
                else
                    fprintf(fp,'ep: %d, batch %d, error: %e\n', epoch, batchId, recerr/(batchId*mbSize));
                    fprintf('ep: %d, batch %d, error: %e\n', epoch, batchId, recerr/(batchId*mbSize));
                end
            end
        end
        %keyboard
        elapsedTime = toc(startTime);
        fprintf(fp,'One full training finished... used %d seconds\n', elapsedTime);
        fprintf('One full training finished... used %d seconds\n', elapsedTime);
    end
    
    if(~isempty(devdata))
        W = gather(bestW);
        visBias = gather(bestVisBias);
        hidBias = gather(bestHidBias);
    else
        W = gather(W);
        visBias = gather(visBias);
        hidBias = gather(hidBias);
    end
    
    fclose(fp);
