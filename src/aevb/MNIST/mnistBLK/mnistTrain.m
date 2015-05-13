load data/batchtraindata.mat
dataTr = single(batchdata);
clear batchdata;
dataTr = dataTr/max(max(dataTr)); % normalization
if(size(dataTr,1)~=784)
    dataTr = dataTr';
end

[NN, NNsetting] = defaultNNsetting(dataTr(:,1:10));
[NN] = defaultNNinit(NN, NNsetting);

[BLK, BLKsetting]=defaultBLKsetting(2); % (K)
[BLK]=defaultBLKinit(BLK, NNsetting.mbSize, NN.D2); % (N, D)

nSamples = size(dataTr,2);
numBatch = floor(nSamples/NNsetting.mbSize);

Loss = zeros(NNsetting.nEpoch,1);

for epoch = 1:NNsetting.nEpoch
    index = randperm(nSamples);
    for batchIdx = 1:numBatch
        fprintf('epoch %d, batchIdx %d\n', epoch, batchIdx);
        % forward propagation
        firstIdx = (batchIdx-1)*NNsetting.mbSize+1;
        lastIdx = batchIdx*NNsetting.mbSize;
        NN = diagFF(NN, NNsetting, dataTr(:,index(firstIdx:lastIdx)));
        Loss(epoch) = Loss(epoch)+NN.loss;
        if(rem(batchIdx,50)==1)
            fprintf(2,'\t recon loss: %f\n', Loss(epoch)/batchIdx/NNsetting.mbSize);
        end
        % sampling BLK model provided hidden representations
        [BLK] = sampleBLK(NN.Z, BLK, BLKsetting);

        % backpropagation
        NN = diagBackProp(NN, NNsetting, BLK);

        % update NN parameter
        NN = updateNN(NN, NNsetting);
    end
    fprintf('epoch %d, log-likelihood is %f\n', epoch, Loss(epoch)/nSamples/NNsetting.L);
    model.W1 = NN.W1;
    model.W2 = NN.W2;
    model.W3 = NN.W3;
    model.W4 = NN.W4;
    model.W5 = NN.W5;
    model.b1 = NN.b1;
    model.b2 = NN.b2;
    model.b3 = NN.b3;
    model.b4 = NN.b4;
    model.b5 = NN.b5;
    model.Mu = BLK.Mu;
    model.Lambda = BLK.Lambda;
    model.cc = BLK.cc;
    model.G = BLK.G;
    model.C = BLK.C;
    % note: need to study / Look_into the details (value, evolution,
    % sensitivity) of BLK prior parameters w.r.t. minibatch_data 
%     save(['model4dNNepoch' num2str(epoch) '.mat'], 'model','Loss');
    save(['model4dBLKNNepoch' num2str(epoch) '.mat'], 'model','Loss');
end
