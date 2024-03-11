#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=stat_fcd_fit
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --output=outputs/stat_fcd_fit.out
#SBATCH --error=outputs/stat_fcd_fit.err

ml MATLAB/R2022b
matlab -nodisplay<<-EOF

clear all;
close all;
addpath ../dynamic_fic_dmf_Cpp Results/ functions/ outputs/ data/
SEED = 1;
%mex ../dynamic_fic_dmf_Cpp/dyn_fic_DMF.cpp
% Load Data
load data/SC_and_5ht2a_receptors.mat
C = 0.2.*sc90./max(sc90(:));
stren = sum(C)./2;
params = dyn_fic_DefaultParams('C',C);
% basic model parameters
params.TR = 2.4;
params.flp = 0.01; 
params.fhi = 0.1; 
params.wsize = 30; 
params.overlap = 28; 
params.N=length(params.C);

load('./data/ts_coma24_AAL_symm_withSC.mat');
% Save in data the timeseries
params.NSUB=13;
indexsub=1:params.NSUB;
for nsub=indexsub
    data(:, :, nsub)=timeseries_CNT24_symm{indexsub};
end
Isubdiag = find(tril(ones(params.N),-1));

params.burnout = 7;
params.T = size(data,2);
params.TMAX = 192 - params.burnout;

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
NHOURS = 16;
% bayesian model params
checkpoint_file = 'Results/stat_fcd/results_v0.mat';
bo_opts = {'IsObjectiveDeterministic',true,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
        'MinWorkerUtilization',4,...
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
ALPHA_range = [0.05 1];
G_range = [0 16];
% seed fixed for a training
params.seed = SEED;
% training
params.fit_fc = true;
params.fit_fcd = false;
%
results = static_fitting(G_range,ALPHA_range,params,bo_opts, emp_fcd);
close all;
% save results
filename = sprintf('Results/stat_fcd_fit/seed_%d.mat',params.seed); % Create filename
save(filename, 'results'); % Save results in a .mat file

EOF
