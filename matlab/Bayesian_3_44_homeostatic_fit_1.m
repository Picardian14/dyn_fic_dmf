#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=BigGHomFit1
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --output=outputs/BigGHomFit1.out
#SBATCH --error=outputs/BigGHomFit1.err

ml MATLAB/R2022b

matlab -nodisplay<<-EOF

clear all;
close all;
addpath ../dynamic_fic_dmf_Cpp Results/ functions/ outputs/ data/
mex ../dynamic_fic_dmf_Cpp/dyn_fic_DMF.cpp
load data/DTI_fiber_consensus_HCP.mat
C = 0.2.*connectivity./max(connectivity(:));
stren = sum(C)./2;
params = dyn_fic_DefaultParams('C',C);
N=length(params.C);

%disp("Amoount of available processes: "+maxNumCompThreads);
%parpool('local', maxNumCompThreads);

bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
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
G_span = 0:0.5:8;

params.obj_rate = 3.44;
for seed=1:100
folder_name = sprintf('Results/Low_G_Range/'); % Create folder name based on seed number
    if ~exist(folder_name, 'dir')
        mkdir(folder_name); % Create folder if it doesn't exist
        fprintf('Folder "%s" created.\n', folder_name);
    else
        fprintf('Folder "%s" already exists.\n', folder_name);
    end
    params.seed=seed;    
    results = findHomeostasis(DECAY_range,LR_range,G_span,params,bo_opts);
    close all;
    filename = sprintf('Results/Low_G_Range/seed_%d.mat',seed); % Create filename
    save(filename, 'results'); % Save results in a .mat file

end
EOF