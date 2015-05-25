function [dataTr, labelTr, faceM, faceSTD] = loadFaceData()
% one extra operation is to renormalize the data

    load ../data/face.mat
    labelTr{1} = single(data(:,end-1));
    labelTr{2} = single(data(:,end));
    
    dataTr = single(data(:,1:end-2));
    clear data;
    dataTr = dataTr/max(max(dataTr));
    dataTr = dataTr';

    % renormalization the data, 
    %   so as to be consistent with the pretrained parameters
    modelPart1 = load('../../../week6/FACE/model/GRBM/model1K.mat');
    faceM = modelPart1.faceM;
    faceSTD = modelPart1.faceSTD;
    
    dataTr = bsxfun(@minus, dataTr, faceM);
    dataTr = bsxfun(@rdivide, dataTr, faceSTD);

end
