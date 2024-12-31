#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=HomFuncEstim
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=4G
#SBATCH --cpus-per-task=12
#SBATCH --output=outputs/HomFuncEstim.out
#SBATCH --error=outputs/HomFuncEstim.err

ml matlab/R2022b

matlab -nodisplay<<-EOF

clear all;
close all;
mex ../dynamic_fic_dmf_Cpp/dyn_fic_DMF.cpp
load SC_and_5ht2a_receptors.mat
C = 0.2.*sc90./max(sc90(:));
stren = sum(C)./2;
params = dyn_fic_DefaultParams('C',C);
N=length(params.C);

disp("Amoount of available processes: "+maxNumCompThreads);
parpool('local', maxNumCompThreads);

total_time = 3600 * 12 % * hours

bo_opts = {'IsObjectiveDeterministic',true,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e15,...
        'MaxTime', total_time,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,...        
        'PlotFcn', {@plotObjectiveModel,@plotMinObjective}};


% Setting model parameters
params.nb_steps = 100000;
params.burnout = 7;
LR_range = [0.1,1000];
DECAY_range = [100,100000];
G_span = 0:0.5:8;


params.obj_rate = 3.44;
seed = 42;
folder_name = sprintf('Results/OBJ_RATE3-44/LongRunWithSeed-%d', seed); % Create folder name based on seed number
if ~exist(folder_name, 'dir')
    mkdir(folder_name); % Create folder if it doesn't exist
    fprintf('Folder "%s" created.\n', folder_name);
else
    fprintf('Folder "%s" already exists.\n', folder_name);
end
params.seed=seed;
it = 1;
results = findHomeostasis(DECAY_range,LR_range,G_span,params,bo_opts);
close all;
filename = sprintf('Results/OBJ_RATE3-44/LongRunWithSeed-%d/iter_%d.mat',seed, it); % Create filename
save(filename, 'results'); % Save results in a .mat file
EOF