clear all;
close all;

load SC_and_5ht2a_receptors.mat
C = 0.2.*sc90./max(sc90(:));
stren = sum(C)./2;
params = dyn_fic_DefaultParams('C',C);
N=length(params.C);


%parpool('local', 16);

bo_opts = {'IsObjectiveDeterministic',true,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,...
        'OutputFcn',@stoppingCriteria,...        %% We will stop when homeostatic fittness reaches less than 0.1
        'PlotFcn', {@plotObjectiveModel,@plotMinObjective}};

% Setting model parameters
params.nb_steps = 100000;
params.burnout = 5;
LR_range = [0.1,1000];
DECAY_range = [100,100000];
% G and OBJ_RATE may change
G_span = 0:0.5:6;

%
params.obj_rate = 3.44;
for seed=11:20
folder_name = sprintf('Results/OBJ_RATE3-44/%d', seed); % Create folder name based on seed number
    if ~exist(folder_name, 'dir')
        mkdir(folder_name); % Create folder if it doesn't exist
        fprintf('Folder "%s" created.\n', folder_name);
    else
        fprintf('Folder "%s" already exists.\n', folder_name);
    end
    params.seed=seed;
    for it=1:20
        results = findHomeostasis(DECAY_range,LR_range,G_span,params,bo_opts);
        close all;
        filename = sprintf('Results/OBJ_RATE3-44/%d/iter_%d.mat',seed, it); % Create filename
        save(filename, 'results'); % Save results in a .mat file
    end
end

params.obj_rate = 6.88;
for seed=1:20
folder_name = sprintf('Results/OBJ_RATE6-88/%d', seed); % Create folder name based on seed number
    if ~exist(folder_name, 'dir')
        mkdir(folder_name); % Create folder if it doesn't exist
        fprintf('Folder "%s" created.\n', folder_name);
    else
        fprintf('Folder "%s" already exists.\n', folder_name);
    end
    params.seed=seed;
    for it=1:20
        results = findHomeostasis(DECAY_range,LR_range,G_span,params,bo_opts);
        close all;
        filename = sprintf('Results/OBJ_RATE6-88/%d/iter_%d.mat',seed, it); % Create filename
        save(filename, 'results'); % Save results in a .mat file
    end
end

params.obj_rate = 1.22;
for seed=1:20
folder_name = sprintf('Results/OBJ_RATE1-22/%d', seed); % Create folder name based on seed number
    if ~exist(folder_name, 'dir')
        mkdir(folder_name); % Create folder if it doesn't exist
        fprintf('Folder "%s" created.\n', folder_name);
    else
        fprintf('Folder "%s" already exists.\n', folder_name);
    end
    params.seed=it;
    for it=1:20
        results = findHomeostasis(DECAY_range,LR_range,G_span,params,bo_opts);
        close all;
        filename = sprintf('Results/OBJ_RATE1-22/%d/iter_%d.mat',seed, it); % Create filename
        save(filename, 'results'); % Save results in a .mat file
    end
end
%%
XatMinObjectiveList344 = zeros(2, 20); % Assuming you have 30 iterations
ObjectiveList344 = zeros(20);
for it = 1:20    
    filename = sprintf('Results/OBJ_RATE3-44/seed_%d.mat', it);       
    loadedData = load(filename);       
    XatMinObjectiveList344(:,it) = [loadedData.results.XAtMinObjective{1,1};loadedData.results.XAtMinObjective{1,2}];
    ObjectiveList344(it) = loadedData.results.MinObjective;
end

XatMinObjectiveList688 = zeros(2, 20); % Assuming you have 30 iterations
ObjectiveList688 = zeros(20);
for it = 1:20    
    filename = sprintf('Results/OBJ_RATE6-88/seed_%d.mat', it);       
    loadedData = load(filename);       
    XatMinObjectiveList688(:,it) = [loadedData.results.XAtMinObjective{1,1};loadedData.results.XAtMinObjective{1,2}];
    ObjectiveList688(it) = loadedData.results.MinObjective;
end

XatMinObjectiveList122 = zeros(2, 20); % Assuming you have 30 iterations
ObjectiveList122 = zeros(20);
for it = 1:20    
    filename = sprintf('Results/OBJ_RATE1-22/seed_%d.mat', it);       
    loadedData = load(filename);       
    XatMinObjectiveList122(:,it) = [loadedData.results.XAtMinObjective{1,1};loadedData.results.XAtMinObjective{1,2}];
    ObjectiveList122(it) = loadedData.results.MinObjective;
end
%%
% Scatter plot 1 with XatMinObjectiveList122 and XatMinObjectiveList344
figure;
scatter(XatMinObjectiveList122(1, :), XatMinObjectiveList122(2,:), [], 'r', 'filled'); % Red for XatMinObjectiveList122
hold on;
scatter(XatMinObjectiveList344(1,:), XatMinObjectiveList344(2,:), [], 'b', 'filled'); % Blue for XatMinObjectiveList344
hold on;
scatter(XatMinObjectiveList688(1,:), XatMinObjectiveList688(2,:), [], 'g', 'filled'); % Blue for XatMinObjectiveList344
title('Optimals with <0.1 fitness');
xlabel('XatMinObjectiveList122 - X-axis');
ylabel('XatMinObjectiveList344 - Y-axis');
legend('1.22Hz', '3.44Hz', '6.88Hz');
hold off;
