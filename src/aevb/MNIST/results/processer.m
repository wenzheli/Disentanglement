models = {
    'model1a.mat';
    'model1A.mat';
    'model1b.mat';
    'model1B.mat';
    'model2a.mat';
    'model2b.mat';
    'model2c.mat';
    'model2d.mat';
    'model3a.mat';
    'model3b.mat';
    'model3c.mat'};

for mid=1:length(models)
    load(models{mid});
    nModels = length(model);
    if(nModels>1)
        for i=1:nModels
            l(i) = model{i}.LL(end,1);
            if(l(i)==0)
                l(i) = -10000;
            end
        end
        [~, i] = max(l);
        model = model{i}; 
    else
        if(iscell(model))
            model = model{1};
        else
            model = model(1);
        end
    end
    
    newname = models{mid};
    newname(1) = 'M';
%     keyboard
    save(newname, 'model');
    clear model;
    clear l;
end