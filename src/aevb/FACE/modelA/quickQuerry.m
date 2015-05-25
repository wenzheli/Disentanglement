% ptZ, unstable, dimension=6
names = dir('modelA_*dim6*epoch20*ptZ*mat');
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

% ptZ, unstable, dimension=6
names = dir('modelA_*dim9*epoch20*ptZ*mat');
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
    fprintf(2,'\ndimension=6, ptZ, minibatch-Sampler\n');
    disp(num2str(losses2));
    [a,b] = min(losses2(:,3));
    disp(names(b).name);
end

