function sampleModelE(modelName)
% sample the observations

    %% part 1. load model & data
    [model, BLK] = loadModelE(modelName);
    
    [dataTr, labelTr, model.faceM, model.faceSTD] = loadFaceData();
    
    %% part 2. calculate the data statistics
    [dataZ, Mdata, Vpost, Vdata, Mprior, Vprior] = inferFF(model, BLK, dataTr);
    plotBlock(Vpost(BLK.order, BLK.order), 'posterior ');
    plotBlock(Vdata(BLK.order, BLK.order), 'data ');
    plotBlock(Vprior(BLK.order, BLK.order), 'prior ');
    
    % visualize the 2D distribution
    plot2D(dataZ, labelTr{1}, model, 'label-1');
    plot2D(dataZ, labelTr{2}, model, 'label-2');
    
    %% part 3. demonstrate the reconstruction capability
    idx = randperm(size(dataTr,2), 225);
    demoRecon(model, dataTr(:,idx));
    
    %% part 4.
    % generage visual samples from Z layer
    [zSparse, zGrid] = sampleLayerZ(Mdata, Vdata);
    
    for blkSample = 1:model.K
%         sampleBlockImg(model, blkSample, zSparse, zGrid);
        sampleMeanImg(model, blkSample, Mdata, zGrid, zSparse, dataZ, labelTr);
    end
end

function [model, BLK] = loadModelE(name)
    load(name);
    
    for blockID = 1:model.K
        model.blocks{blockID} = find(BLK.cc==blockID);
    end
end

function [Z, Mdata, Vpost, Vdata, Mprior, Vprior] = inferFF(model, blk, dataTr)
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
    
    Mdata = zeros(D2,1);
    Vpost = zeros(D2, D2);
    Vdata = zeros(D2, D2);
    Mprior = zeros(D2, 1);
    Vprior = zeros(D2, D2);

    Z = zeros(D2,N);
    Beta = zeros(D2^2, batchSize);
    Sigma = zeros(D2, D2, batchSize);

    for firstIdx=1:batchSize:N
        lastIdx = min(N, firstIdx+batchSize-1);
        mbSize = lastIdx-firstIdx+1;
        h1 = sigmoid(bsxfun(@plus, W1'*dataTr(:,firstIdx:lastIdx), b1));
        Z(:,firstIdx:lastIdx) = bsxfun(@plus, W2'*h1, b2);
        Beta = exp(bsxfun(@plus, W3'*h1, b3));
        Sigma = reshape(Beta, [D2, D2, mbSize]);

        Mdata = Mdata + sum(Z,2)/N;
        Vpost = Vpost + sum(Sigma,3)/N;
    end

    Vdata = cov(Z');
    Vprior = blk.Sigma;
    Mprior = blk.Mu;
    
    fprintf(2,'mean of the hidden layer: data vs prior\n');
    disp(Mdata(blk.order));
    disp(Mprior(blk.order));
    
    fprintf(2,'covariance of the hidden layer: data vs posterior vs prior\n');
    disp(Vdata(blk.order,blk.order));
    disp(Vpost(blk.order,blk.order));
    disp(Vprior(blk.order,blk.order));
    
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
    
    for blockID = 1:model.K
        if(isempty(model.blocks{blockID}))
            continue;
        elseif(length(model.blocks{blockID})==1)
            id1 = model.blocks{blockID}(1);
            id2 = model.blocks{blockID}(1);
        else
            id1 = model.blocks{blockID}(1);
            id2 = model.blocks{blockID}(2);
        end
        subplot(1,model.K,blockID);
        if(length(unique(label))>10)
            gscatter(Z(id1, :), Z(id2, :), label,  '', '>*do.','','off')
        else
            gscatter(Z(id1, :), Z(id2, :), label,  '', '>*do.')
        end
        title([ titleStr ' block ' num2str(blockID)], 'FontSize',12,'FontWeight','Demi');
    end
end

function [zSparse, zGrid] = sampleLayerZ(M, V)
% sample N datapoints in the the Z layer space
%   which will be used in the (sampling + visualization) function


    X = mvnrnd(M, V, 5000);
    X = sortrows(X, 1:size(V,1));
    zSparse = X(250:500:5000,:);
    zGrid= X(251:20:4750, :);
    
end

function sampleBlockImg(model, blkSample, zSparse, zGrid)

    if(~isempty(model.blocks{blkSample}))

        M = size(zSparse,1);
        K = model.K;

        dimImg = sqrt(model.D0);
        img = zeros(15*(dimImg+1)-1, 15*(dimImg+1)-1);
        for m = 1:M
            z = zSparse(m,:)';
            for i=1:15
                for j=1:15
                    n = (i-1)*15+j;
                    z(model.blocks{blkSample}) = zGrid(n,model.blocks{blkSample})';%zGrid(n, model.blocks{blkSample})';
                    
                    h2 = sigmoid(model.W4'*z + model.b4);

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
end

function sampleMeanImg(model, blkSample, zMean, zGrid, zSparse, dataZ, dataLabel)
% generate images that sample the blkSample-th block in detail
%   fixing the coordinates of other blocks to be mean values

    if(~isempty(model.blocks{blkSample}))
        % enumerate M images
        nBlocks = model.K;
        dimImg = sqrt(model.D0);
        img = zeros(15*(dimImg+1)-1, 15*(dimImg+1)-1);

        z = zMean(:);

        for i=1:15
            for j=1:15
                n = (i-1)*15+j;

                z(model.blocks{blkSample}) = zGrid(n,model.blocks{blkSample})';
                h2 = sigmoid(model.W4'*z + model.b4);

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

        scatterImage(dataZ, dataLabel{1}, model, zSparse, zGrid, img, blkSample);
    end
end

function scatterImage(Z, label, model, zSparse, zGrid, img, blkID)
% zSparse: sparse sample coordinates from block "blkid"
% zGrid: grid sample coordinates from block "blkid"
% img: visualization of this block
    if(~isempty(model.blocks{blkID}))
        H = figure;
        set(H, 'Position',[100,100,1400,600]);
        % left: labeled scatter plot as background, 
        %   visualize the Grid coordinates
        %   visualize the Sparse coordinates

        if(length(model.blocks{blkID})==1)
            id1 = model.blocks{blkID}(1);
            id2 = model.blocks{blkID}(1);
        else
            id1 = model.blocks{blkID}(1);
            id2 = model.blocks{blkID}(2);
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
end

function X = sigmoid(X)
    X = 1./(1+exp(-X));
end

function demoRecon(model, dataTr)
% generate a 15x15 original image set
% generate a 15x15 reconstructed image set
    h1 = sigmoid(bsxfun(@plus, model.W1'*dataTr, model.b1));
    z = bsxfun(@plus, model.W2'*h1, model.b2);
    
    h2 = sigmoid(bsxfun(@plus, model.W4'*z, model.b4));
                
    if(strcmp(model.data, 'mnist'))
        x = 1./(1+exp(-bsxfun(@plus, model.W5'*h2, model.b5)));
    else
        x = bsxfun(@plus, model.W5'*h2, model.b5);
        if(strcmp(model.data, 'face'))
            x = bsxfun(@times, x, model.faceSTD);
            x = bsxfun(@plus, x, model.faceM);
            dataTr = bsxfun(@times, dataTr, model.faceSTD);
            dataTr = bsxfun(@plus, dataTr, model.faceM);
        end
    end
    
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