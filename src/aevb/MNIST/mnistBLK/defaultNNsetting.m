function [NN, setting]= defaultNNsetting(dataTr)

    NN.D0 = size(dataTr,1);
    NN.D1 = 200;
    NN.D2 = 4;
    NN.D3 = 200;
    setting.L = 10;
    setting.mbSize = 50;

    NN.X1      = zeros(NN.D0, setting.mbSize);
    NN.h1      = zeros(NN.D1, setting.mbSize);
    NN.mu      = zeros(NN.D2, setting.mbSize);
    NN.beta	= zeros(NN.D2, setting.mbSize);
    NN.lambda 	= zeros(NN.D2, setting.mbSize);
    NN.Z       = zeros(NN.D2, setting.mbSize*setting.L);
    NN.h2      = zeros(NN.D3, setting.mbSize*setting.L);
    NN.Y       = zeros(NN.D0, setting.mbSize*setting.L);
    NN.X2      = zeros(NN.D0, setting.mbSize*setting.L);

    setting.sgdParams = [
        1/1000 0.9;
        1/2000 0.9;
        1/3000 0.5];
    setting.alg = 'sgd';
    NN.lrate = setting.sgdParams(1,1);
    NN.momentum = setting.sgdParams(1,2);

    setting.nEpoch = 20;
end
