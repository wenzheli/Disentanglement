names = dir('modelA_dim6_1000*SGD*mat');
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

names = dir('modelA_dim4_1000*SGD*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
    %     fprintf('model: ');
    %     disp(names(i).name);
        losses2(i,1) = model.rho;
        losses2(i,2) = model.const;
        losses2(i,3) = model.ftLoss(end)/65960;
    %     disp(num2str(losses(i)));
    end
    fprintf(2,'\ndimension=4, ptZ, minibatch-Sampler\n');
    disp(num2str(losses2));
    [a,b] = min(losses2(:,3));
    disp(names(b).name);
end

names = dir('modelA_dim6_500*SGD*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
    %     fprintf('model: ');
    %     disp(names(i).name);
        losses3(i,1) = model.rho;
        losses3(i,2) = model.const;
        losses3(i,3) = model.ftLoss(end)/65960;
    %     disp(num2str(losses(i)));
    end
    fprintf(2,'\ndimension=6, ptZ, minibatch-Sampler\n');
    disp(num2str(losses3));
    [a,b] = min(losses3(:,3));
    disp(names(b).name);
end

names = dir('modelA_dim4_500*SGD*mat');
if(~isempty(names))
    nModels = length(names);
    for i=1:nModels
        load(names(i).name);
    %     fprintf('model: ');
    %     disp(names(i).name);
        losses4(i,1) = model.rho;
        losses4(i,2) = model.const;
        losses4(i,3) = model.ftLoss(end)/65960;
    %     disp(num2str(losses(i)));
    end
    fprintf(2,'\ndimension=4, ptZ, minibatch-Sampler\n');
    disp(num2str(losses4));
    [a,b] = min(losses4(:,3));
    disp(names(b).name);
end

