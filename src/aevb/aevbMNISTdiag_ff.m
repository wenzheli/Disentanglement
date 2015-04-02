% feed-forward propagation to 
% 1. check the reconstructed samples
% 2. visualize the hidden space

load data/batchtraindata.mat;
dataTr = batchdata';
clear batchdata;

load MNISTdiag.mat
model = model{1};   % select model with lowest reconstruction error 

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

%% part 0. feed-forward propagation, encoding and then decoding for reconstruction
N = 10000;
firstIdx = 1;
lastIdx = N;

X1 = dataTr(:, firstIdx:lastIdx);
h1 = tanh(bsxfun(@plus, W1'*X1, b1));
mu = bsxfun(@plus, W2'*h1, b2);
beta = bsxfun(@plus, W3'*h1, b3);
Z = mu;
h2 = tanh(bsxfun(@plus, W4'*Z, b4));
Y = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5))); 

%% part 1. select 100 samples to compare the original and reconstructed images
img0 = zeros(29*10-1,29*10-1);  % original image
img1 = zeros(29*10-1,29*10-1);  % reconsructed image
id = 1;
for i=1:10
    for j=1:10
        x = X1(:,id);
        x = reshape(x,[28,28]);
        img0((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x';

        x = Y(:,id);
        x = reshape(x,[28,28]);
        img1((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x';

        id=id+1;
    end
end
figure, subplot(1,2,1), imshow(img0,[]);
title('original', 'FontSize',12','FontWeigth','Demi');
subplot(1,2,2), imshow(img1,[]);
title('reconstruct', 'FontSize',12','FontWeigth','Demi');

%% part 2. visualize the hidden space according to digit classes
Zs = cell(10,1);
for i=1:10
    Zs{i} = Z(:,batchlabel(1:10000)==i);
end

figure, plot(Zs{1}(1,:), Zs{1}(2,:),'*')
hold on, plot(Zs{2}(1,:), Zs{2}(2,:),'r*')
hold on, plot(Zs{3}(1,:), Zs{3}(2,:),'g*')
hold on, plot(Zs{4}(1,:), Zs{4}(2,:),'k*')
hold on, plot(Zs{5}(1,:), Zs{5}(2,:),'c*')
hold on, plot(Zs{6}(1,:), Zs{6}(2,:),'m*')
hold on, plot(Zs{7}(1,:), Zs{7}(2,:),'y*')
hold on, plot(Zs{8}(1,:), Zs{8}(2,:),'gd')
hold on, plot(Zs{9}(1,:), Zs{9}(2,:),'rd')
hold on, plot(Zs{10}(1,:), Zs{10}(2,:),'kd')
hleg = legend('digit0','digit1','digit2','digit3','digit4','digit5','digit6','digit7','digit8','digit9');
set(hleg, 'Location', 'SouthEast', 'FontSize',14, 'FontWeight','Demi');

