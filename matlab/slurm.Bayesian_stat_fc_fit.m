#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=BigGRangeShortAlpha_stat_fc
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --output=outputs/BigGRangeShortAlpha_stat_fc.out
#SBATCH --error=outputs/BigGRangeShortAlpha_stat_fc.err

ml MATLAB/R2022b
matlab -nodisplay<<-EOF

clear all;
close all;
addpath ../dynamic_fic_dmf_Cpp Results/ functions/ outputs/ data/
sub_experiment_name = "BigGRangeShortAlpha";
%mex ../dynamic_fic_dmf_Cpp/dyn_fic_DMF.cpp
% Load Data
load data/SC_and_5ht2a_receptors.mat
C = 0.2.*sc90./max(sc90(:));
stren = sum(C);
params = dyn_fic_DefaultParams('C',C);
% fitting options
params.fit_fc = true;
params.fit_fcd = false;
% type of fic calculation
params.with_plasticity=false;
params.with_decay=false;

% Setting model parameters
params.return_rate=true;
params.return_fic=true;
params.return_bold=true;

params.obj_rate = 3.44;

% basic model parameters
params.TR = 2.4;
params.flp = 0.008; 
params.fhi = 0.08; 
params.wsize = 30; 
params.overlap = 28; 
params.N=length(params.C);
%params.seed = sub_experiment_name;
load('./data/ts_coma24_AAL_symm_withSC.mat');
% Save in data the timeseries
params.NSUB=13;
indexsub=1:params.NSUB;
for nsub=indexsub
    data(:, :, nsub)=timeseries_CNT24_symm{indexsub};
end
Isubdiag = find(tril(ones(params.N),-1));
params.burnout = 10;
params.T = size(data,2);
params.TMAX = 192 - params.burnout;

indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;    
    Wdata(:,:,nsub)=data(:, params.burnout:end, nsub) ; 
    WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);    
    WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end
WFCdataF = permute(WFCdataF, [2,3,1]);
emp_fc = mean(WFCdataF,3);
NHOURS = 12;
% bayesian model params
checkpoint_file = "Results/stat_fc/results_"+sub_experiment_name+".mat";
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
        'MinWorkerUtilization',8,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',NHOURS*3600,...  
        'OutputFcn', @saveToFile,...
        'SaveFileName', checkpoint_file,...
        'PlotFcn', {@plotObjectiveModel,@plotMinObjective}};
        %'OutputFcn',@stoppingCriteria,...        %% We leave it running without criteria



params.win_start = 0:params.wsize-params.overlap:params.TMAX-params.wsize-1;
params.nwins = length(params.win_start);
params.nb_steps = fix((params.T)*params.TR)/params.dtt; % Generate the same amount of time points ant then remove the transient period
ALPHA_range = [0.65 0.85];
G_range = [0 2.5];
params.ALPHA_range = ALPHA_range;
params.G_range = G_range;
% training

%%
results = static_fitting(G_range,ALPHA_range,params,bo_opts, emp_fc);
close all;
% save results
filename = sprintf('Results/stat_fc/%s.mat',sub_experiment_name); % Create filename
save(filename, 'results'); % Save results in a .mat file

EOF
