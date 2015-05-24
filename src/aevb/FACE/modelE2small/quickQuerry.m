% ptZ, unstable, dimension=6
names = dir('modelE_*dim6*epoch20*ptZ*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
    %     fprintf('model: ');
    %     disp(names(i).name);
        losses1(i,1) = model.rho;
        losses1(i,2) = model.const;
        losses1(i,3) = model.ftLoss(end)/65960;
    %     disp(num2str(losses(i)));
    end
    fprintf(2,'\ndimension=6, ptZ, minibatch-Sampler\n');
    disp(num2str(losses1));
    [a,b] = min(losses1(:,3));
    disp(names(b).name);
end

% ptZ, unstable, dimension=9
names = dir('modelE_*dim9*epoch20*ptZ*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
        losses2(i,1) = model.rho;
        losses2(i,2) = model.const;
        losses2(i,3) = model.ftLoss(end)/65960;
    end
    fprintf(2,'\ndimension=9, ptZ, minibatch-Sampler\n');
    disp(num2str(losses2));
    [a,b] = min(losses2(:,3));
    disp(names(b).name);
end

%% ptZ, unstable, dimension=12
names = dir('modelE_*dim12*epoch20*ptZ*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
        losses3(i,1) = model.rho;
        losses3(i,2) = model.const;
        losses3(i,3) = model.ftLoss(end)/65960;
    end
    fprintf(2,'\ndimension=12, ptZ, minibatch-Sampler\n');
    disp(num2str(losses3));
    [a,b] = min(losses3(:,3));
    disp(names(b).name);
end

%% ptH1, unstable, dimension=6
names = dir('modelE_*dim6*epoch20*ptH1*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
        losses4(i,1) = model.rho;
        losses4(i,2) = model.const;
        losses4(i,3) = model.ftLoss(end)/65960;
    end
    fprintf(2,'\ndimension=6, ptH1, minibatch-Sampler\n');
    disp(num2str(losses4));
    [a,b] = min(losses4(:,3));
    disp(names(b).name);
end

%% ptH1, unstable, dimension=9
names = dir('modelE_*dim9*epoch20*ptH1*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
        losses5(i,1) = model.rho;
        losses5(i,2) = model.const;
        losses5(i,3) = model.ftLoss(end)/65960;
    end
    fprintf(2,'\ndimension=9, ptH1, minibatch-Sampler\n');
    disp(num2str(losses5));
    [a,b] = min(losses5(:,3));
    disp(names(b).name);
end

%% ptH1, unstable, dimension=12
names = dir('modelE_*dim12*epoch20*ptH1*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
        losses6(i,1) = model.rho;
        losses6(i,2) = model.const;
        losses6(i,3) = model.ftLoss(end)/65960;
    end
    fprintf(2,'\ndimension=12, ptH1, minibatch-Sampler\n');
    disp(num2str(losses6));
    [a,b] = min(losses6(:,3));
    disp(names(b).name);
end








% ptZ, unstable, dimension=6
names = dir('modelEmemo_*dim6*epoch20*ptZ*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
    %     fprintf('model: ');
    %     disp(names(i).name);
        losses7(i,1) = model.rho;
        losses7(i,2) = model.const;
        losses7(i,3) = model.ftLoss(end)/65960;
    %     disp(num2str(losses(i)));
    end
    fprintf(2,'\ndimension=6, ptZ, average-Sampler\n');
    disp(num2str(losses7));
    [a,b] = min(losses7(:,3));
    disp(names(b).name);
end

% ptZ, unstable, dimension=9
names = dir('modelEmemo_*dim9*epoch20*ptZ*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
        losses8(i,1) = model.rho;
        losses8(i,2) = model.const;
        losses8(i,3) = model.ftLoss(end)/65960;
    end
    fprintf(2,'\ndimension=9, ptZ, average-Sampler\n');
    disp(num2str(losses8));
    [a,b] = min(losses8(:,3));
    disp(names(b).name);
end

%% ptZ, unstable, dimension=12
names = dir('modelEmemo_*dim12*epoch20*ptZ*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
        losses9(i,1) = model.rho;
        losses9(i,2) = model.const;
        losses9(i,3) = model.ftLoss(end)/65960;
    end
    fprintf(2,'\ndimension=12, ptZ, average-Sampler\n');
    disp(num2str(losses9));
    [a,b] = min(losses9(:,3));
    disp(names(b).name);
end