#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=G_LR_Freqs
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --output=outputs/G_LR_Freqs.out
#SBATCH --error=outputs/G_LR_Freqs.err

ml MATLAB/R2022b

matlab -nodisplay<<-EOF


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
G_span = 0:0.5:12;
%%
params.obj_rate = 3.44;
fitResult= load("./Results/LogFit344.mat").fitResult;
LR_range = logspace(-1,2,100);
num_freqs = 1000;
all_PSD = zeros(length(LR_range),length(G_span), num_freqs, N);
all_freqs = zeros(length(LR_range),length(G_span), num_freqs);


for idx_LR=1:length(LR_range)
    parfor idx_G=1:length(G_span)
        thispars = params;
        thispars.G = G_span(idx_G);
        thispars.lrj = LR_range(idx_LR);
        thispars.taoj = fitResult.a*thispars.lrj^fitResult.b
        [rates, rates_inh] = dyn_fic_DMF(thispars, thispars.nb_steps,'rate');
        
    end
end

save("./Results/PSD_G_LR.max","all_PSD");
save("./Results/freqs_G_LR.max","all_freqs");
EOF