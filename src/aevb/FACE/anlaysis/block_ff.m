% a quick evaluation of the model
% 1. reconstruction capability
% 2. latent space visualization
% 3. statistics of latent space

%% part 0: load model and preprocess to extract basic model information
nBlocks = 2;
D2 = 2;

load ../model/ft2B.mat

% preprocess 2: extract the information of each block
W5 = model.W5;  b5 = model.b5;
W4 = model.W4;  b4 = model.b4;
W3 = model.W3;  b3 = model.b3;
W2 = model.W2;  b2 = model.b2;
W1 = model.W1;  b1 = model.b1;

bSize = zeros(nBlocks,1);
for blockID = 1:nBlocks
    bSize(blockID) = size(W4{blockID},1);
end

%% part 1. feed-forward propagation, encoding and then decoding for reconstruction
load ../data/face.mat
dataTr = single(data(:,1:end-2));
clear data;
dataTr = dataTr/max(max(dataTr)); % normalization
dataTr = dataTr';

ptModel = load('../data/faceStats.mat');
dataTr = bsxfun(@minus, dataTr, ptModel.faceM);
dataTr = bsxfun(@rdivide, dataTr, ptModel.faceSTD);

N = size(dataTr,2);%10000;
firstIdx = 1;
lastIdx = N;

X1 = dataTr(:, firstIdx:lastIdx);
h1 = 1./(1+exp(-bsxfun(@plus, W1'*X1, b1)));
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

blockID = 1;
h2 = (bsxfun(@plus, W4{blockID}'*Z{blockID}, b4));
for blockID=2:nBlocks
    h2 = h2+W4{blockID}'*Z{blockID};
end
h2 = 1./(1+exp(-h2));

Y = bsxfun(@plus, W5'*h2, b5); 

z = Z{1};
for blockID = 2:nBlocks
    z = [z; Z{blockID}];
end
M = mean(z,2);
Vdata = cov(z');
Vtrue = zeros(D2*nBlocks);
% for blockID = 1:nBlocks
%     Vtrue((blockID-1)*D2+1:blockID*D2, (blockID-1)*D2+1:blockID*2) = mean(at{blockID},3);
% end
% ms = cell(nBlocks,1);
% vs = cell(nBlocks,1);
% for blockID = 1:nBlocks
%     ms{blockID} = mean(Z{blockID},2);
%     vs{blockID} = cov(Z{blockID}');
% end

%% part 2. select 100 samples to compare the original and reconstructed images
dim = 64;
img0 = zeros((dim+1)*10-1,(dim+1)*10-1);
img1 = zeros((dim+1)*10-1,(dim+1)*10-1);
X1 = bsxfun(@times, X1, ptModel.faceSTD);
X1 = bsxfun(@plus, X1, ptModel.faceM);
Y = bsxfun(@times, Y, ptModel.faceSTD);
Y = bsxfun(@plus, Y, ptModel.faceM);

id = 1;
for i=1:10
    for j=1:10
        x = X1(:,id*10);
        x = reshape(x,[dim,dim]);
        img0((i-1)*(dim+1)+1:i*(dim+1)-1,(j-1)*(dim+1)+1:j*(dim+1)-1) = x;

        x = Y(:,id*10);
        x = reshape(x,[dim,dim]);
        img1((i-1)*(dim+1)+1:i*(dim+1)-1,(j-1)*(dim+1)+1:j*(dim+1)-1) = x;
        id=id+1;
    end
end
H1 = figure;
set(H1, 'Position',[100,100,1000,400]);
subplot(1,2,1), imshow(img0,[]);
title('original', 'FontSize',12','FontWeight','Demi');
subplot(1,2,2), imshow(img1,[]);
title('reconstruct', 'FontSize',12','FontWeight','Demi');

% Zs = cell(10,nBlocks);
% m = cell(1,nBlocks);
% v = cell(1,nBlocks);
% for blockID = 1:nBlocks
%     for i=1:10
%         Zs{i, blockID} = Z{blockID}(:,batchlabel(1:10000)==i);
%     end
%     m{blockID} = mean(Z{blockID},2);
%     v{blockID} = cov(Z{blockID}');
% end
% 
% % 
% H2 = figure;
% set(H2, 'Position',[100,100,1000,400]);
% for blockID = 1:nBlocks
%     subplot(1,nBlocks,blockID), plot(Zs{1,blockID}(1,:), Zs{1, blockID}(2,:),'*');
%     hold on, plot(Zs{2 ,blockID}(1,:), Zs{2 ,blockID}(2,:),'r*')
%     hold on, plot(Zs{3 ,blockID}(1,:), Zs{3 ,blockID}(2,:),'g*')
%     hold on, plot(Zs{4 ,blockID}(1,:), Zs{4 ,blockID}(2,:),'k*')
%     hold on, plot(Zs{5 ,blockID}(1,:), Zs{5 ,blockID}(2,:),'c*')
%     hold on, plot(Zs{6 ,blockID}(1,:), Zs{6 ,blockID}(2,:),'m*')
%     hold on, plot(Zs{7 ,blockID}(1,:), Zs{7 ,blockID}(2,:),'y*')
%     hold on, plot(Zs{8 ,blockID}(1,:), Zs{8 ,blockID}(2,:),'gd')
%     hold on, plot(Zs{9 ,blockID}(1,:), Zs{9 ,blockID}(2,:),'rd')
%     hold on, plot(Zs{10 ,blockID}(1,:), Zs{10 ,blockID}(2,:),'kd')
%     title(['block ' num2str(blockID)], 'FontSize',12,'FontWeight','Demi')
% end
% 
% %save(['statBlock' num2str(nBlocks) '.mat'], 'm','v','M','V');
% 
