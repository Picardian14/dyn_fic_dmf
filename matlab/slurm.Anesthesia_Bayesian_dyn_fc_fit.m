#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=HCP_Deep_sameranges_12hs_dyn_fc
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --chdir=/network/iss/home/ivan.mindlin/dyn_fic_dmf/matlab
#SBATCH --output=outputs/HCP_Deep_sameranges_12hs_dyn_fc.out
#SBATCH --error=outputs/HCP_Deep_sameranges_12hs_dyn_fc.err

ml matlab/R2022b
matlab -nodisplay<<-EOF

clear all;
close all;
addpath ../dynamic_fic_dmf_Cpp Results/ functions/ outputs/ data/

folder_name = 'Results/dyn_fc';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

sub_experiment_name = "HCP_Deep_sameranges_12hs";
%mex ../dynamic_fic_dmf_Cpp/dyn_fic_DMF.cpp
% Load Data
load data/DTI_fiber_consensus_HCP.mat
C = connectivity(1:200,1:200);
C = 0.2.*C./max(C(:));
params = dyn_fic_DefaultParams('C',C);
% Setting model parameters
% training
params.fit_fc = true;
params.fit_fcd = false;
% tpye of fic calculation
params.with_decay=true;
params.with_plasticity=true;
% return type
params.return_bold=true;
params.return_rate=true;
params.return_fic=true;
params.obj_rate = 3.44;
% basic model parameters
params.TR = 2;
params.flp = 0.01; 
params.fhi = 0.1; 
params.wsize = 30; 
params.overlap = 29; 
params.N=length(params.C);

load('./data/BOLD_timeseries_Deep.mat')
% Save in data the timeseries
params.NSUB=length(BOLD_timeseries_Deep);
indexsub=1:params.NSUB;
for nsub=indexsub
    data(:, :, nsub)=BOLD_timeseries_Deep{nsub}(1:200,:);
end
Isubdiag = find(tril(ones(params.N),-1));

params.burnout = 7;
params.T = size(data,2);
params.TMAX = params.T - params.burnout;

indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;    
    Wdata(:,:,nsub)=data(:, params.burnout:end, nsub) ; 
    WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);
    WFCdata(nsub,:,:)=corrcoef(squeeze(Wdata(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
    WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end

WFCdata = permute(WFCdata, [2,3,1]);
WFCdataF = permute(WFCdataF, [2,3,1]);
emp_fc = mean(WFCdataF,3);
NHOURS = 12;
% bayesian model params
checkpoint_file = "Results/dyn_fc/results_"+sub_experiment_name+".mat";
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
        'MinWorkerUtilization',4,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',NHOURS*3600,...  
        'OutputFcn', @saveToFile,...
        'SaveFileName', checkpoint_file,...
        'PlotFcn', {@plotObjectiveModel,@plotMinObjective}};
        %'OutputFcn',@stoppingCriteria,...        %% We leave it running without criteria





%params.win_start = np.arange(0, TMAX - wsize - 1, wsize - overlap)
params.win_start = 0:params.wsize-params.overlap:params.TMAX-params.wsize-1;
params.nwins = length(params.win_start);
%int((data.shape[-1]-burnout)*params['TR']/params['dtt'])

params.nb_steps = fix((params.T)*params.TR)/params.dtt; % Generate the same amount of time points ant then remove the transient period
LR_range = [1 1000];
G_range = [0.1 16];
params.LR_range = LR_range;
params.G_range = G_range;
% seed fixed for a training
%params.seed = sub_experiment_name;

results = dynamic_fitting(G_range,LR_range,params,bo_opts, emp_fc);
close all;
% save results

filename = sprintf('%s/%s.mat', folder_name, sub_experiment_name); % Create filename
save(filename, 'results'); % Save results in a .mat file

EOF