function [curr_e,curr_i,J_t] = dmf_deco18_dynamic_fic(obj_rate,l_rate,j_decay,inic,Tmax,dt,...
    I0,Jexte,Jexti,w,JN,C,Jini,we,...
    gamma,sigma,taog,taon,wgaine,wgaini,...
    receptors,g_e,g_i,Ie,Ii,ce,ci)
% Implements Deco 2018 Curr Biol Dynamic Mean Field Model
%
% INPUTS
% inic = seed of the random simulator. If inic=0 uses current seed.
% Tmax = simulation time points
% dt = integration step [ms]
% I0 = external current [nA]
% Jexte = external->E coupling
% Jexti = external->I coupling
% w = local excitatory recurrence
% JN = excitatory synaptic coupling [nA]
% C = connectivity matrix, should fit length of receptors
% J = feedback inhibitory control parameter to be estimated
% we = Global Coupling parameter
% gamma = Kinetic Parameter of Excitation
% sigma = amplitude of noise [nA]
% taog = GABA time constant
% taon = NMDA time constant
% wgaine = excitatory gain modulation
% wgaini = inhibitory gain modulation
% receptors = receptor density per region
% g_e = excitatory conductance
% g_i = same than above but inhibitory
% Ie = excitatory threshold for nonlineariy
% Ii = same than above but inhibitory
% ce = excitatory non linear shape parameter
% ci = same than above but inhibitory
%
% OUTPUTS
% curr_e = currents of excitatory pools per region
% curr_i = currents of inhibitory pools per region

% 
if inic~=0
    rng(inic,'twister');
end

% Initial parameters
N=size(C,1); % number of units

% Model simulations
curr_e=zeros(Tmax/10,N);
curr_i = curr_e;
% records every 10 time points
% s_e = zeros(Tmax/10,N);
% s_i = s_e;
J_t = curr_i;
% net_i = curr_e;
% loc_i = curr_e;
sn=0.001*ones(N,1);
sg=0.001*ones(N,1);
cont = 0;
tid = 1;
% Jini = ones(N,1);
J=Jini;
% j_offset = 0.0025;
for t=1:Tmax  % integration in time
    
    xn=I0*Jexte+w*JN*sn+we*JN*C*sn-J.*sg;% excitatory currents
    xg=I0*Jexti+JN*sn-sg;% inhibitory currents    
%     net_i(t,:) = we*JN*C*sn;
%     loc_i(t,:) = I0*Jexte+w*JN*sn-J.*sg;
    rn = curr2rate(xn,wgaine,g_e,Ie,ce,receptors); % excitatory population rate function        
    rg = curr2rate(xg,wgaini,g_i,Ii,ci,receptors);% inhibitory population rate function    
    sn=sn+dt*(-sn/taon+(1-sn)*gamma.*rn./1000.)+sqrt(dt)*sigma*randn(N,1);
    sn(sn>1) = 1;
    sn(sn<0) = 0;
    sg=sg+dt*(-sg/taog+rg./1000.)+sqrt(dt)*sigma*randn(N,1);
    sg(sg>1) = 1;
    sg(sg<0) = 0;
    % updating local inhibition
    J = J + dt*(l_rate.*rg.*(rn-obj_rate)./(10.^6) - J./j_decay);
%     J = J + dt*(l_rate./1000.*rg.*(rn-obj_rate));
    
    % storing variables
    cont = cont+1;
    if cont==10
%         s_e(tid,:) = sn;
%         s_i(tid,:) = sg;
        curr_e(tid,:) = xn;
        curr_i(tid,:) = xg;
        J_t(tid,:) = J;
        tid = tid +1;
        cont = 0;
    end
        
end