% the struction of the model
% q(z|x): dim-784 Xs --> dim-200 hu --> dim-2 Gaussian Z
% p(x|z): dim-2 Gaussian Z --> dim-200 hu --> dim-784 Xs

load MNISTdiag.mat
model = model{1};  %select the model with the smallest reconstruction error

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

% it is easy to sample images when there are only 2 nodes in the hidden space
% the only task is to sample z1 and z2

% heuristic sampling:
z1 = -10:3:20;
z2 = -10:3:20;
z1 = [z1 -3.5:0.75:11]; % finer-grained sampling around the mean
z2 = [z2 -3.5:0.75:11]; 
z1 = sort(unique(z1));
z2 = sort(unique(z2));
N1 = length(z1);
N2 = length(z2);

% there could be other sampling method, e.g. sampling z ~ E_x[q(z|x)]

img = zeros(29*N1-1,29*N2-1);
id = 1;
for i=1:N1
    for j=1:N2
        Z = [z1(i); z2(j)];
        h2 = tanh(bsxfun(@plus, W4'*Z, b4));
        x = 1./(1+exp(-bsxfun(@plus, W5'*h2, b5))); 
        x = reshape(x,[28,28]);
        img((i-1)*29+1:i*29-1,(j-1)*29+1:j*29-1) = x';
        id=id+1;
    end
end

figure, imshow(img, []);
