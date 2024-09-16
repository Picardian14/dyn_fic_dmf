clear all;
load('matlab/data/SC_and_5ht2a_receptors.mat')
C = sc90;
% Normalize C
C = 0.2 * C / max(C(:));
% Set parameters
params = dyn_fic_DefaultParams('C', C);
%params.seed = 1;
params.TR = 2.4;
params.G = 2.11;
params.J = 0.75 * params.G * sum(params.C, 1).' + 1;
params.flp = 0.01;
params.fhp = 0.1;
nb_steps = 448800;
params.with_plasticity = true;
params.with_decay = true;
params.return_bold = true;
params.return_rate = false;
params.return_fic = true;
burnout = 8;
all_sim_fc = zeros(12,90,90);
parfor idx=1:12    
    [rates, rates_inh, BOLD, fic_t] = dyn_fic_DMF(params, nb_steps);
    BOLD = BOLD(:,burnout:end);
    filt_bold = filter_bold(BOLD', params.flp, params.fhp, params.TR);        
    all_sim_fc(idx, :, :)=corrcoef(filt_bold);
end
%save('python/Results/ALL_matlab_dynfic.mat', 'BOLD', 'rates', 'rates_inh', 'fic_t', 'filt_bold');

