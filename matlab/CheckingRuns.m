
clear all;
close all;
addpath ../dynamic_fic_dmf_Cpp Results/ functions/ outputs/ data/
sub_experiment_name = "GoodRange_2";
%mex ../dynamic_fic_dmf_Cpp/dyn_fic_DMF.cpp
% Load Data
load data/DTI_fiber_consensus_HCP.mat
C = 0.2.*connectivity(1:200,1:200)./max(connectivity(1:200,1:200));
params = dyn_fic_DefaultParams('C',C);
stren = sum(params.C);
% FITTING PARAMS
params.fit_fc = true;
params.fit_fcd = false;


% TYPE OF FIC CALC
params.with_plasticity=false;
params.with_decay=false;

% OUTPUT model parameters
params.return_rate=true;
params.return_fic=true;
params.return_bold=true;
params.obj_rate = 3.44;


%% basic model parameters
params.TR = 2;
params.flp = 0.01; 
params.fhi = 0.1; 
params.wsize = 30; 
params.overlap = 29; 
params.N=length(params.C);

load('./data/BOLD_timeseries_Awake.mat')
% Save in data the timeseries
params.NSUB=length(BOLD_timeseries_Awake);
indexsub=1:params.NSUB;
for nsub=indexsub
    data(:, :, nsub)=BOLD_timeseries_Awake{nsub}(1:200,:);
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
checkpoint_file = "Results/dyn_fcd/results_"+sub_experiment_name+".mat";
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





%params.win_start = np.arange(0, TMAX - wsize - 1, wsize - overlap)
params.win_start = 0:params.wsize-params.overlap:params.TMAX-params.wsize-1;
params.nwins = length(params.win_start);
%int((data.shape[-1]-burnout)*params['TR']/params['dtt'])

params.nb_steps = fix((params.T)*params.TR)/params.dtt; % Generate the same amount of time points ant then remove the transient period
LR_range = [1 1000];
G_range = [0.1 16];
% seed fixed for a training
%params.seed = sub_experiment_name;
% training

coeffs = load("data/LinearFitCoefficients.mat");
a = coeffs.a;
b = coeffs.b;
%%
params.G = 1.60; % 2.9 for minObjective

params.alpha = 0.72; % 0.76

params.lrj = 4.58;                          
params.taoj = exp(a+log(params.lrj)*b);
% save a safe copy to send to dyn_fic function
if params.with_plasticity
    params.J = 0.75*params.G*stren' + 1; % updates it
else
    params.J = params.alpha*params.G*stren' + 1; % updates it
end
%% Running
[rates, rates_inh, bold, fic_t] = dyn_fic_DMF(params, params.nb_steps);

%% Processing
% takeout transient simulation
rates = rates(:, (params.burnout*params.TR/params.dtt):end);
%all_rates(idx, :) = mean(rates, 2);
bold = bold(:,params.burnout:end); % remove initial transient
bold(isnan(bold))=0;
bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
if isempty(bold)      
    disp("G: "+params.G+" LR: "+params.lrj+" Gave empty bold");
    out_error = nan;
    return
end
% Filtering and computing FC
filt_bold = filter_bold(bold',params.flp,params.fhi,params.TR);
isubfc = find(tril(ones(params.N),-1));
if params.fit_fc
    sim_fc = corrcoef(filt_bold);
    disp("calculo FC FC")
    %all_sim_fc(idx, :, :) = sim_fc;
elseif params.fit_fcd      
    disp("calculo fcd")
    sim_fcd = compute_fcd(filt_bold,params.wsize,params.overlap,isubfc);
    sim_fcd(isnan(sim_fcd))=0;
    sim_fcd = corrcoef(sim_fcd);
    if isempty(sim_fcd)                    
        sim_fcd = zeros(size(sim_fcd));
        return
    end
    %all_sim_fcd(idx, :, :) = sim_fcd;
else
    disp('error: target observable not set')
    out_error=nan; 
end

if params.fit_fc
    %mean_fc = mean(all_sim_fc, 1);
    % SOLO ESTOY COMPARANDO CON 1 FC
    disp(" corr")
    out_error = 1-corr2(sim_fc(isubfc),emp_fc(isubfc));
elseif params.fit_fcd             
    % SOLO ESTOY COMPARANDO CON 1 FC      
    try
        [~,~,out_error] = kstest2(sim_fcd(:),emp_fcd(:));
    catch E
        disp(E)
        disp("G: "+ params.G);
        disp("LR: "+ params.lrj);
        if isempty(sim_fcd)                    
            disp("FCD Was empty ");
        end
        out_error=1;
    end
else
    disp('error: target observable not set')
    out_error=nan;      
end
%mean_rates = mean(all_rates,1);
mean_rates = mean(rates,2);
outdata = {mean_rates};
%end