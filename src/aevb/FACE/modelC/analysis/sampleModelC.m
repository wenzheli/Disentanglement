function sampleBlockModel(modelName)
% sample the observations

    %% part 1. load model & data
    model = loadModel(modelName);
    [dataTr, labelTr, model.faceM, model.faceSTD] = loadFaceData();
    
    %% part 2. calculate the data statistics
    [dataZ, Mdata, Vpost, Vdata, Mprior, Vprior] = inferFF(model, dataTr);
    plotBlock(Vpost, 'posterior ');
    plotBlock(Vdata, 'data ');
    plotBlock(Vprior, 'prior ');
    
    % visualize the 2D distribution
    plot2D(dataZ, labelTr{1}, model, 'label-1');
    plot2D(dataZ, labelTr{2}, model, 'label-2');
    
    %% part 3. demonstrate the reconstruction capability
    idx = randperm(size(dataTr,2), 225);
    demoRecon(model, dataTr(:,idx));
    
    %% part 4.
    % generage visual samples from Z layer
    keyboard
    [zSparse, zGrid] = sampleLayerZ(Mdata, Vdata, model);
    
    for blkSample = 1:model.nBlocks
%         sampleBlockImg(model, blkSample, zSparse, zGrid, Mdata);
        sampleMeanImg(model, blkSample, Mdata, zGrid, zSparse, dataZ, labelTr);
    end
end

function model = loadModel(name)
    load(name);
    % if model is structure (i.e. isstruct(model)), 
    %   further processing is not needed
    if(iscell(model.W2))
        model.nBlocks = length(model.W2);
    end
    
    % if diagonal posterior model, convert to block model
    if(strcmp(model.shape, 'diag'))
        model = diagBlockConvert(model);
    end
    
    nBlocks = model.nBlocks;
    Ns = zeros(nBlocks,1);
    for blockID = 1:nBlocks
        Ns(blockID) = length(model.blocks{blockID});
    end
    model.Ns = cumsum([0; Ns]);
    
end

function NN = diagBlockConvert(NN)
%   size of W:      [DimIn, DimOut]
%   size of bias:   [Dimout, 1]
    
    nBlocks = max(NN.cc);
    NN.nBlocks = nBlocks;
    
    W2 = NN.W2; NN.W2 = [];
    W3 = NN.W3; NN.W3 = [];
    W4 = NN.W4; NN.W4 = [];
    b2 = NN.b2; NN.b2 = [];
    b3 = NN.b3; NN.b3 = [];

    for blkid = 1:nBlocks
        NN.blocks{blkid} = find(NN.cc==blkid);
    end

    for blkid=1:nBlocks
        NN.W2{blkid} = W2(:, NN.blocks{blkid});
        NN.b2{blkid} = b2(NN.blocks{blkid});
        
        NN.W3{blkid} = W3(:, NN.blocks{blkid});
        NN.b3{blkid} = b3(NN.blocks{blkid});
        
        NN.W4{blkid} = W4(NN.blocks{blkid},:);
    end     
end

function [Z, Mdata, Vpost, Vdata, Mprior, Vprior] = inferFF(model, dataTr)
% Vdata: covariance of dataset in the hidden layer space
    W3 = model.W3;
    W2 = model.W2;
    W1 = model.W1;

    b3 = model.b3;
    b2 = model.b2;
    b1 = model.b1;
    
    D2 = model.D2;
    
    N = size(dataTr,2);
    batchSize = N;%100;
    
    Z = [];
    Mdata = [];
    Vpost = [];
    Vdata = [];
    Mprior = [];
    Vprior = [];
    
    if(strcmp(model.shape, 'block'))
        NW = model.NW;
        
        Z = cell(model.nBlocks,1);
        Beta = cell(model.nBlocks,1);
        Sigma = cell(model.nBlocks,1);
        Mdata = zeros(D2*model.nBlocks,1);
        Vpost = zeros(D2*model.nBlocks, D2*model.nBlocks);
        Vdata = zeros(D2*model.nBlocks, D2*model.nBlocks);
        Mprior = zeros(D2*model.nBlocks,1);
        Vprior = zeros(D2*model.nBlocks,D2*model.nBlocks);
        
        for blockID = 1:model.nBlocks
            Z{blockID} = zeros(D2,N);
            Beta{blockID} = zeros(D2^2, batchSize);
            Sigma{blockID} = zeros(D2, D2, batchSize);
        end
        
        for blockID = 1:model.nBlocks
            for firstIdx=1:batchSize:N
                lastIdx = min(N, firstIdx+batchSize-1);
                mbSize = lastIdx-firstIdx+1;
                h1 = sigmoid(bsxfun(@plus, W1'*dataTr(:,firstIdx:lastIdx), b1));
                Z{blockID}(:,firstIdx:lastIdx) = bsxfun(@plus, W2{blockID}'*h1, b2{blockID});
                Beta{blockID} = exp(bsxfun(@plus, W3{blockID}'*h1, b3{blockID}));
                Sigma{blockID} = reshape(Beta{blockID}, [D2, D2, mbSize]);
                
                Mdata((blockID-1)*D2+1:blockID*D2) = ...
                    Mdata((blockID-1)*D2+1:blockID*D2) + sum(Z{blockID},2)/N;
                Vpost((blockID-1)*D2+1:blockID*D2, (blockID-1)*D2+1:blockID*D2) = ...
                    Vpost((blockID-1)*D2+1:blockID*D2, (blockID-1)*D2+1:blockID*D2) + ...
                    sum(Sigma{blockID},3)/N;
            end
            
        end
        tmpZ = Z{1};
        for blkID = 2:model.nBlocks
            tmpZ = [tmpZ; Z{blkID}];
        end
        Vdata = cov(tmpZ');
        
        NW = updateNW(Z, NW);
        for blockID = 1:model.nBlocks
            Mprior((blockID-1)*D2+1:blockID*D2) = NW.mu{blockID};
            Vprior((blockID-1)*D2+1:blockID*D2, (blockID-1)*D2+1:blockID*D2) = ...
                inv(NW.Lambda{blockID});
        end
    elseif(strcmp(model.shape, 'diag')) 
        
        Z = zeros(D2, N);
        
        beta = zeros(D2, batchSize);
        sigma = zeros(D2, batchSize);
        Mdata = zeros(D2, batchSize);
        Vpost = zeros(D2, batchSize);
        
        for firstIdx=1:batchSize:N
            lastIdx = min(N, firstIdx+batchSize-1);

            h1 = sigmoid(bsxfun(@plus, W1'*dataTr(:,firstIdx:lastIdx), b1));
            Z(:,firstIdx:lastIdx) = bsxfun(@plus, W2'*h1, b2);
            beta = exp(bsxfun(@plus, W3'*h1, b3));
            sigma = exp(beta);

            Mdata = Mdata + sum(Z,2)/N;
            Vpost = Vpost + sum(sigma,s)/N;
        end
        Vdata = cov(Z');
        
    else
        error('the posterior should be either "block" or "diag"');
    end
    
    fprintf(2,'mean of the hidden layer: data vs prior\n');
    disp(Mdata);
    disp(Mprior);
    
    fprintf(2,'covariance of the hidden layer: data vs posterior vs prior\n');
    disp(Vdata);
    disp(Vpost);
    disp(Vprior);
    
end

function plotBlock(V, titleStr)
% visualize the correlation matrix
% left: abs(covariance matrix)
% right: abs(correlation matrix)
    s = sqrt(diag(V));
    C = V./(s*s');
    
    d = size(C,1);
    
    while(d<1000)
        d = d*2;
    end
    
    C = imresize(C, [d,d], 'nearest');
    V = imresize(V, [d,d], 'nearest');
    
    figure, subplot(1,2,1), imshow(abs(V), []);
    title([titleStr 'covariance'], 'FontSize',12,'FontWeight','Demi');
    subplot(1,2,2), imshow(abs(C), []);
    title([titleStr 'correlation'], 'FontSize',12,'FontWeight','Demi');
end

function plot2D(Z, label, model, titleStr)
    H = figure;
    set(H, 'Position',[100,100,1200,500]);
    
    for blockID = 1:model.nBlocks
        if(length(model.blocks{blockID})==1)
            id1 = 1;
            id2 = 1;
        else
            id1 = 1;
            id2 = 2;
        end
        subplot(1,model.nBlocks,blockID);
        if(length(unique(label))>10)
            gscatter(Z{blockID}(id1, :), Z{blockID}(id2, :), label,  '', '>*do.','','off')
        else
            gscatter(Z{blockID}(id1, :), Z{blockID}(id2, :), label,  '', '>*do.')
        end
        title([ titleStr ' block ' num2str(blockID)], 'FontSize',12,'FontWeight','Demi');
    end
end

function [zSparse, zGrid] = sampleLayerZ(M, V, model)
% sample N datapoints in the the Z layer space
%   which will be used in the (sampling + visualization) function

    nBlocks = model.nBlocks;
    Ns = zeros(nBlocks,1);
    for blockID = 1:nBlocks
        Ns(blockID) = length(model.blocks{blockID});
    end
    Ns = cumsum([0; Ns]);
    
    X = mvnrnd(M, V, 5000);
    X = sortrows(X, 1:size(V,1));
    %zSparse = X(125:250:5000,:);
    zSparse = X(250:500:5000,:);
    zGrid = cell(nBlocks, 1);
    for blockID=1:nBlocks
        zGrid{blockID} = X(251:20:4750,model.blocks{blockID});       % 225=15^2 samples
    end

end

function sampleBlockImg(model, blkSample, zSparse, zGrid, zMean)
% generate images that sample the blkSample-th block in detail
%   fixing the coordinates of other blocks

    % enumerate M images
    M = size(zSparse,1);
    
    nBlocks = model.nBlocks;
    dimImg = sqrt(model.D0);
    img = zeros(15*(dimImg+1)-1, 15*(dimImg+1)-1);
    
    for m = 1:M
        for blk = 1:nBlocks
            z{blk} = zSparse(m, model.blocks{blk})';
%             zMean = mean(zSparse)';
%             z{blk} = zMean(model.blocks{blk});
        end
        for i=1:15
            for j=1:15
                n = (i-1)*15+j;
                
                z{blkSample} = zGrid{blkSample}(n,:)';%zGrid(n, model.blocks{blkSample})';
                h2 = model.W4{1}'*z{1} + model.b4;
                for blk=2:nBlocks
                    h2 = model.W4{blk}'*z{blk} + h2;
                end
                h2 = sigmoid(h2);
                
                if(strcmp(model.data, 'mnist'))
                    x = 1./(1+exp(-bsxfun(@plus, model.W5'*h2, model.b5)));
                else
                    x = bsxfun(@plus, model.W5'*h2, model.b5);
                    if(strcmp(model.data, 'face'))
                        x = x.*model.faceSTD;
                        x = x+model.faceM;
                    end
                end
                
                img((i-1)*(dimImg+1)+1:i*(dimImg+1)-1,(j-1)*(dimImg+1)+1:j*(dimImg+1)-1) = ...
                    reshape(x, [dimImg, dimImg]);
            end
        end
        figure, imshow(img,[]);
        title(['enumerate plot: block ' num2str(blkSample) ' sampled'], 'FontSize', 10);
        
    end
end

function sampleMeanImg(model, blkSample, zMean, zGrid, zSparse, dataZ, dataLabel)
% generate images that sample the blkSample-th block in detail
%   fixing the coordinates of other blocks to be mean values

    % enumerate M images
    nBlocks = model.nBlocks;
    dimImg = sqrt(model.D0);
    img = zeros(15*(dimImg+1)-1, 15*(dimImg+1)-1);
    
    zMean = zMean(:);
    for blk = 1:nBlocks
        z{blk} = zMean(model.blocks{blk});
    end
    
    for i=1:15
        for j=1:15
            n = (i-1)*15+j;

            z{blkSample} = zGrid{blkSample}(n,:)';%(n, model.blocks{blkSample})';
            h2 = model.W4{1}'*z{1} + model.b4;
            for blk=2:nBlocks
                h2 = model.W4{blk}'*z{blk} + h2;
            end
            h2 = sigmoid(h2);
            
            if(strcmp(model.data, 'mnist'))
                x = 1./(1+exp(-bsxfun(@plus, model.W5'*h2, model.b5)));
            else
                x = bsxfun(@plus, model.W5'*h2, model.b5);
                if(strcmp(model.data, 'face'))
                    x = x.*model.faceSTD;
                    x = x+model.faceM;
                end
            end

            img((i-1)*(dimImg+1)+1:i*(dimImg+1)-1,(j-1)*(dimImg+1)+1:j*(dimImg+1)-1) = ...
                reshape(x, [dimImg, dimImg]);
        end
    end
    
    figure, imshow(img,[]);
    title(['mean plot: block ' num2str(blkSample) ' sampled'], 'FontSize', 12', 'FontWeight', 'Demi');
    
    scatterImage(dataZ{blkSample}, dataLabel{1}, model, zSparse(:, model.blocks{blkSample}), zGrid{blkSample}, img, blkSample);
end

function scatterImage(Z, label, model, zSparse, zGrid, img, blkID)
% zSparse: sparse sample coordinates from block "blkid"
% zGrid: grid sample coordinates from block "blkid"
% img: visualization of this block

    H = figure;
    set(H, 'Position',[100,100,1400,600]);
    % left: labeled scatter plot as background, 
    %   visualize the Grid coordinates
    %   visualize the Sparse coordinates
    
    if(length(model.blocks{blkID})==1)
        id1 = 1;
        id2 = 1;
    else
        id1 = 1;
        id2 = 2;
    end
    
    subplot(1,2,1),
    if(length(unique(label))>10)
        gscatter(Z(id1,:), Z(id2,:), label,  'byg', '*.','','off')
    else
        gscatter(Z(id1,:), Z(id2,:), label,  'byg', '*.')
    end
    
    hold on, 
    plot(zGrid(:, id1), zGrid(:, id2), 'ro', 'MarkerSize',6,'LineWidth',2);
    plot(zSparse(:, id1), zSparse(:, id2), 'kd', 'MarkerSize',6,'LineWidth',2);
    
    subplot(1,2,2),
    imshow(img,[]);
    title(['block ' num2str(blkID) ], 'FontSize',12,'FontWeight','Demi');
end

function X = sigmoid(X)
    X = 1./(1+exp(-X));
end

function demoRecon(model, dataTr)
% generate a 15x15 original image set
% generate a 15x15 reconstructed image set
    for blockID = 1:model.nBlocks
        h1 = sigmoid(bsxfun(@plus, model.W1'*dataTr, model.b1));
        z{blockID} = bsxfun(@plus, model.W2{blockID}'*h1, model.b2{blockID});
    end
    
    h2 = bsxfun(@plus, model.W4{1}'*z{1}, model.b4);
    for blk=2:model.nBlocks
        h2 = model.W4{blk}'*z{blk} + h2;
    end
    h2 = sigmoid(h2);
                
    if(strcmp(model.data, 'mnist'))
        x = 1./(1+exp(-bsxfun(@plus, model.W5'*h2, model.b5)));
    else
        x = bsxfun(@plus, model.W5'*h2, model.b5);
        if(strcmp(model.data, 'face'))
            x = bsxfun(@times, x, model.faceSTD);
            x = bsxfun(@plus, x, model.faceM);
        end
    end
    
    dataTr = bsxfun(@times, dataTr, model.faceSTD);
    dataTr = bsxfun(@plus, dataTr, model.faceM);
    
    % visualize the images
    dimImg = sqrt(model.D0);
    imgOrig = zeros((dimImg+1)*15-1, (dimImg+1)*15-1);
    imgRecon = zeros((dimImg+1)*15-1, (dimImg+1)*15-1);
    for i=1:15
        for j=1:15
            n = (i-1)*15+j;
            imgOrig((i-1)*(dimImg+1)+1:i*(dimImg+1)-1, (j-1)*(dimImg+1)+1:j*(dimImg+1)-1) = ...
                reshape(dataTr(:,n), dimImg, dimImg);
                        
            imgRecon((i-1)*(dimImg+1)+1:i*(dimImg+1)-1, (j-1)*(dimImg+1)+1:j*(dimImg+1)-1) = ...
                reshape(x(:,n), dimImg, dimImg);
        end
    end
    
    H = figure;
    set(H, 'Position', [100, 100, 1200, 500]);
    subplot(1,2,1), imshow(imgOrig, []);
    subplot(1,2,2), imshow(imgRecon, []);
end

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
    modelPart1 = load('../data/model1K.mat');
    faceM = modelPart1.faceM;
    faceSTD = modelPart1.faceSTD;
    
    dataTr = bsxfun(@minus, dataTr, faceM);
    dataTr = bsxfun(@rdivide, dataTr, faceSTD);

end
