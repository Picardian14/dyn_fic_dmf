#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=HCP_Deep_sameranges_stat_fcd_fit
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --output=outputs/HCP_Deep_sameranges_stat_fcd_fit.out
#SBATCH --error=outputs/HCP_Deep_sameranges_stat_fcd_fit.err

ml matlab/R2022b
matlab -nodisplay<<-EOF

clear all;
close all;
addpath ../dynamic_fic_dmf_Cpp Results/ functions/ outputs/ data/

folder_name = 'Results/stat_fcd';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

sub_experiment_name = "HCP_Deep_sameranges";

% Load Data
load data/DTI_fiber_consensus_HCP.mat
C = 0.2.*connectivity(1:200,1:200)./max(connectivity(1:200,1:200));
stren = sum(C)./2;
params = dyn_fic_DefaultParams('C',C);
% Fitting params
params.fit_fc = false;
params.fit_fcd = true;
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
    tmp_time_fc = compute_fcd(WdataF(:,:,nsub)',params.wsize, params.overlap,Isubdiag);
    emp_fcd(nsub, :, :) = corrcoef(tmp_time_fc);
end
WFCdata = permute(WFCdata, [2,3,1]);
WFCdataF = permute(WFCdataF, [2,3,1]);
emp_fc = mean(WFCdataF,3);
NHOURS = 24;
% bayesian model params
checkpoint_file = "Results/stat_fcd/results_"+sub_experiment_name+".mat";
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

% Setting model parameters
params.return_rate=true;
params.return_fic=true;
params.return_bold=true;
params.with_plasticity=false;
params.with_decay=false;
params.obj_rate = 3.44;



%params.win_start = np.arange(0, TMAX - wsize - 1, wsize - overlap)
params.win_start = 0:params.wsize-params.overlap:params.TMAX-params.wsize-1;
params.nwins = length(params.win_start);
%int((data.shape[-1]-burnout)*params['TR']/params['dtt'])

params.nb_steps = fix((params.T)*params.TR)/params.dtt; % Generate the same amount of time points ant then remove the transient period
ALPHA_range = [0.01 0.99];
G_range = [0.1 16];
% seed fixed for a training
%params.seed = sub_experiment_name;
% train
%
results = static_fitting(G_range,ALPHA_range,params,bo_opts, emp_fcd);
close all;
% save results

filename = sprintf('%s/%s.mat', folder_name, sub_experiment_name); % Create filename
save(filename, 'results'); % Save results in a .mat file

EOF