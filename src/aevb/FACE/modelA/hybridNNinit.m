function NN = hybridNNinit(NN, setting)
% initialize the network using partially pretrained parameters
%   1. the parameters between observations and neighboring hidden layer
%   are pretrained
%   2. the parameters of the PCA part is initialized randomly

    %% firstly randomly initialize the parameters
    NN = defaultNNinit(NN, setting);
    
    %% secondly, revise the first/last layer parameters from pretrained model
    namePart1 = '/home/dong/Research/DeepLearning/Projects/RepLearning/week6/FACE/model/GRBM/model1K.mat';
    modelPart1 = load(namePart1);
    NN.faceM = modelPart1.faceM;
    NN.faceSTD = modelPart1.faceSTD;
    
    NN.W1 = modelPart1.W1;
    NN.b1 = modelPart1.hidBias1;
    NN.W5 = NN.W1';
    NN.b5 = modelPart1.visBias1;
    
end