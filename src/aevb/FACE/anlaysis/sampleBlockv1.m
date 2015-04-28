function [img1, img2] = sampleBlock(name)
    load(['../model/' name '.mat']);
    nBlocks = 2;
    D2 = 2;
    nModels = size(model,1)*size(model,2);
    if(nModels>1)
        for i=1:nModels
            l(i) = model{i}.LL(end,1);
            if(l(i)==0)
                l(i) = -10000;
            end
        end
        [~, mid] = max(l);
        model = model{mid}; 
    else
        model = model(1);
    end
    
    % part 2. load data and estimate the marginal posterior distribution
    load ../data/face.mat
    dataTr = single(data(:,1:end-2));
    clear data;
    dataTr = dataTr/max(max(dataTr)); % normalization
    dataTr = dataTr';

    ptModel = load('../data/faceStats.mat');
    dataTr = bsxfun(@minus, dataTr, ptModel.faceM);
    dataTr = bsxfun(@rdivide, dataTr, ptModel.faceSTD);

    [ms,vs,Mdata,Vdata,Vtrue] = aevbStat(model, dataTr);
    
    perturbUnit = sqrt(diag(Vdata))/10;

    % part 3. sample images: firstly sample z then sample x|z
    % sample each blocks respectively
    s = 6000;
    rv = mvnrnd(Mdata, Vdata, s);
    rv = sort(rv);
    for blockID = 1:nBlocks
        z{blockID} = rv(200:400:end,(blockID-1)*D2+1:blockID*D2);
        N(blockID) = size(z{blockID},1);
    end
    
    %% start sampling
    W5 = model.W5;
    W4 = model.W4;

    b5 = model.b5;
    b4 = model.b4;

    img1 = cell(3,1); % block 1 fixed
    id = 1;
    dim = 64;
    imgLabel1 = zeros(1,2);
    for i=2:5:N(1)
        for j=2:5:N(1)
            img1{id} = zeros((dim+1)*N(2)-1,(dim+1)*N(2)-1);
            for k=1:N(2)
                for l=1:N(2)
                    Z1 = [z{1}(i,1); z{1}(j,2)] + (rand(2,1)-0.5).*perturbUnit(1:2);
                    Z2 = [z{2}(k,1); z{2}(l,2)];
                    h2 = 1./(1+exp(-(W4{1}'*Z1 + W4{2}'*Z2 + b4)));
                    x = bsxfun(@plus, W5'*h2, b5);
                    
                    x = x.*ptModel.faceSTD;
                    x = x+ptModel.faceM;
                    
                    x = reshape(x,[dim,dim]);
                    img1{id}((k-1)*(dim+1)+1:k*(dim+1)-1,(l-1)*(dim+1)+1:l*(dim+1)-1) = x;
                end
            end
            id=id+1;
            imgLabel1(id,:) = Z1';
        end
    end
   
    for i=1:length(img1)
        figure, imshow(img1{i},[]);
        title(['Z(1,2): (' num2str(imgLabel1(i,1)) ', ' num2str(imgLabel1(i,2)) ')'], 'FontSize',12,'FontWeight','Demi'); 
    end
    
    img2 = cell(3,1); % block 1 fixed
    id = 1;
    imgLabel2 = zeros(1,2);
    for i=2:5:N(2)
        for j=2:5:N(2)
            img2{id} = zeros((dim+1)*N(1)-1,(dim+1)*N(1)-1);
            for k=1:N(1)
                for l=1:N(1)
                    Z2 = [z{2}(i,1); z{2}(j,2)] + (rand(2,1)-0.5).*perturbUnit(3:4);
                    Z1 = [z{1}(k,1); z{1}(l,2)];
                    h2 = 1./(1+exp(-(W4{1}'*Z1 + W4{2}'*Z2 + b4)));
                    x = bsxfun(@plus, W5'*h2, b5);
                    
                    x = x.*ptModel.faceSTD;
                    x = x+ptModel.faceM;
                    
                    x = reshape(x,[dim,dim]);
                    img2{id}((k-1)*(dim+1)+1:k*(dim+1)-1,(l-1)*(dim+1)+1:l*(dim+1)-1) = x;
                end
            end
            id=id+1;
            imgLabel2(id,:) = Z2';
        end
    end
    for i=1:length(img2)
        figure, imshow(img2{i},[]);
        title(['Z(3,4): (' num2str(imgLabel2(i,1)) ', ' num2str(imgLabel2(i,2)) ')'], 'FontSize',12,'FontWeight','Demi'); 
    end
    
    
end

function [ms, vs, Mdata, Vdata, Vtrue] = aevbStat(model, dataTr)
    
    W5 = model.W5;
    W4 = model.W4;
    W3 = model.W3;
    W2 = model.W2;
    W1 = model.W1;

    b5 = model.b5;
    b4 = model.b4;
    b3 = model.b3;
    b2 = model.b2;
    b1 = model.b1;
    
    D2 = length(b2{1});

    if(iscell(W2))
        nBlocks = length(W2);
    else
        nBlocks = 1;
    end

    %% part 0. feed-forward propagation, encoding and then decoding for reconstruction
    N = 1000;
    firstIdx = 1;
    lastIdx = N;

    X1 = dataTr(:, firstIdx:lastIdx);
    h1 = 1./(1+exp(-(bsxfun(@plus, W1'*X1, b1))));
    for blockID = 1:nBlocks
        mu{blockID} = bsxfun(@plus, W2{blockID}'*h1, b2{blockID});
        beta{blockID} = bsxfun(@plus, W3{blockID}'*h1, b3{blockID});
        beta{blockID} = exp(beta{blockID});
        at{blockID} = reshape(beta{blockID}, [D2, D2, N]);
        Z{blockID} = mu{blockID};
    end
    for i=1:N
        at{blockID}(:,:,i) = at{blockID}(:,:,i)*at{blockID}(:,:,i)';
    end
    Vtrue = zeros(D2*nBlocks);
    for blockID = 1:nBlocks
        Vtrue((blockID-1)*D2+1:blockID*D2, (blockID-1)*D2+1:blockID*2) = mean(at{blockID},3);
    end
    
    Zs = cell(10,nBlocks);
    ms = cell(1,nBlocks);
    vs = cell(1,nBlocks);
    for blockID = 1:nBlocks
        ms{blockID} = mean(Z{blockID},2);
        vs{blockID} = cov(Z{blockID}');
    end

    z = Z{1}';
    for blockID = 2:nBlocks
        z = [z Z{blockID}'];
    end
    % full mean and covariance matrix
    Mdata = mean(z);
    Vdata = cov(z);
end
