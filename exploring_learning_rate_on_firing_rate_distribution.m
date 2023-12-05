%% Exploring effect of the learning rate of inhibitory plasticity rule 
% on the currents and firing rate distribution
load SC_and_5ht2a_receptors.mat
C=sc90/max(sc90(:))*0.2;
% load mean5HT2A_bindingaal.mat mean5HT2A_aalsymm % loads receptor density
rec2a=receptors./max(receptors);
N=90;

% Set General Model Parameters
pars.dt=0.1; % ms
pars.taon=100; % NMDA tau ms
pars.taog=10; % GABA tau ms
pars.gamma=0.641; % Kinetic Parameter of Excitation
pars.sigma=0.01; % Noise SD nA
pars.JN=0.15; % excitatory synaptic coupling nA
pars.I0=0.382; % effective external input nA
pars.Jexte=1.; % external->E coupling
pars.Jexti=0.7;% external->I coupling
pars.w=1.4; % local excitatory recurrence
pars.de=0.16; % excitatory conductance
pars.Ie=0.403; % excitatory threshold for nonlineariy
pars.g_e=310.; % excitatory non linear shape parameter
pars.di=0.087;
pars.Ii=0.288;
pars.g_i=615.;
pars.wgaine=0; % neuromodulatory gain
pars.wgaini=0; % neuromodulatory gain
stren = sum(C);
%% Running simulation
obj_rate = 3.44;
j_decay = 4000;
l_rate = 10;
G0=4; % ~best fit to empirical rsfMRI data
j_alpha = 0.75;
Jn = G0.*j_alpha*stren + 1;
pars.tmax=300000; % time points in dt (0.1ms)
pars.burnout = 20000; % ms
pars.iniconds = 2;%rand();
mean_fic_dist = zeros(N,1);

% Running with plastic FIC
tic

[curr_e,curr_i,fic_t] = dmf_deco18_dynamic_fic(obj_rate,l_rate,j_decay,pars.iniconds,...
    pars.tmax,pars.dt,pars.I0,pars.Jexte,...
    pars.Jexti,pars.w,pars.JN,C,...
    Jn',G0,...
    pars.gamma,pars.sigma,pars.taog,...
    pars.taon,pars.wgaine,pars.wgaini,...
    rec2a,pars.g_e,pars.g_i,pars.Ie,...
    pars.Ii,pars.de,pars.di);
% removing first points
curr_e = curr_e(pars.burnout:end,:);
curr_i = curr_i(pars.burnout:end,:);
fic_t = fic_t(pars.burnout:end,:);

frs_e = (curr2rate_whole_rec(curr_e,pars.wgaine,pars.g_e,pars.Ie,pars.de,rec2a)); % exct firing rates
frs_i = (curr2rate_whole_rec(curr_i,pars.wgaini,pars.g_i,pars.Ii,pars.di,rec2a)); % inh firing rates

% Fitting distribution to extract mean
for n=1:N
    data = fic_t(:,n);
%     data(data<=0) = eps;
%     pd = fitdist(data,'loglogistic');
%     mean_fic_dist(n) = mean(pd);
    mean_fic_dist(n) = mean(data);
    

end

% Running with fix FIC
% thisJn = mean_fic_dist;
thisJn = Jn';
[curr_e_fix,curr_i_fix] = dmf_deco18_debug(pars.iniconds,...
    pars.tmax,pars.dt,pars.I0,pars.Jexte,...
    pars.Jexti,pars.w,pars.JN,C,...
    thisJn,G0,...
    pars.gamma,pars.sigma,pars.taog,...
    pars.taon,pars.wgaine,pars.wgaini,...
    rec2a,pars.g_e,pars.g_i,pars.Ie,...
    pars.Ii,pars.de,pars.di);
% removing first points
curr_e_fix = curr_e_fix(pars.burnout:end,:);
curr_i_fix = curr_i_fix(pars.burnout:end,:);

frs_e_fix = (curr2rate_whole_rec(curr_e_fix,pars.wgaine,pars.g_e,pars.Ie,pars.de,rec2a)); % exct firing rates
frs_i_fix = (curr2rate_whole_rec(curr_i_fix,pars.wgaini,pars.g_i,pars.Ii,pars.di,rec2a)); % inh firing rates

toc

% Plotting distribution
figfold = '/media/ruben/ssd240/Matlab/cb-neuromod-master/dmf_learning_fic/statistics_per_region/';
realT = pars.tmax/10 - pars.burnout +1;
% rand_reg = randperm(N,1);
% sel_t = realT-5000:realT;
sel_t = 1:realT;
xxvals = sel_t/1000; 
for rand_reg=2%N
    
    figfile = ['fix_',num2str(j_alpha ),'_vs_dyn_fic_statistics_lr_',num2str(l_rate),'_tao_',num2str(j_decay),'_G_',num2str(G0),'_reg_',num2str(rand_reg)];
    figure('units','normalized','outerposition',[0 0 1 1],'PaperOrientation','landscape','visible','on'); % Maximize figure.
    
    subplot(3,3,1:3)
    % yyaxis left
    p1=plot(xxvals,frs_e_fix(sel_t,rand_reg));hold on
    p2=plot(xxvals,frs_e(sel_t,rand_reg));
    plot([xxvals(1) xxvals(end)],[mean(frs_e(:,rand_reg)) mean(frs_e(:,rand_reg))],...
        'color',get(p2,'color')*0.8,'linewidth',2);
    plot([xxvals(1) xxvals(end)],[mean(frs_e_fix(:,rand_reg)) mean(frs_e_fix(:,rand_reg))],...
        'color',get(p1,'color')*0.8,'linewidth',2);
    l=legend('Fix','Dyn');
    xlabel('Time (s)')
    ylabel('r^E (Hz)')
    grid on
    
    yyaxis right
    p3=plot(xxvals,fic_t(sel_t,rand_reg),'linewidth',2);
    ylabel('FIC')
    xlim([xxvals(1) xxvals(end)])
    set(l,'String',{'Fix','Dyn'})
    
    title({['Fix Mean FR = ',num2str(mean(frs_e_fix(:,rand_reg))),...
        ', Dyn Mean FR = ',num2str(mean(frs_e(:,rand_reg)))],...
        ['G = ',num2str(G0),', lr = ',num2str(l_rate),', \tau = ',num2str(j_decay)]})
    
    subplot(3,3,4)
    histogram(curr_e_fix(:,rand_reg),'edgecolor','none');hold on
    histogram(curr_e(:,rand_reg),'edgecolor','none');hold on
    legend('Fix','Dyn')
    xlabel('I^E (nA)')
    grid on
    
    
    subplot(3,3,7)
    histogram(curr_i_fix(:,rand_reg),'edgecolor','none');hold on
    histogram(curr_i(:,rand_reg),'edgecolor','none');hold on
    legend('Fix','Dyn')
    xlabel('I^I (nA)')
    grid on
    
    
    subplot(3,3,5)
    h1=histfit(frs_e_fix(:,rand_reg),[],'gamma');hold on    
    h2=histfit(frs_e(:,rand_reg),[],'gamma');hold on
    
    set(h1(1),'edgecolor','none','facealpha',0.6)
    set(h1(2),'color',get(h1(1),'facecolor'),'linewidth',2)
    set(h2(1),'edgecolor','none','facealpha',0.6,'facecolor',[0.8500 0.3250 0.0980])
    set(h2(2),'color',get(h2(1),'facecolor'),'linewidth',2)
    
    legend([h1(2),h2(2)],'Fix, gamma fit','Dyn gamma fit')
    xlabel('r^E (Hz)')
    grid on
    
    
    subplot(3,3,8)
    histogram(frs_i_fix(:,rand_reg),'edgecolor','none');hold on
    histogram(frs_i(:,rand_reg),'edgecolor','none');hold on
    legend('Fix','Dyn')
    xlabel('r^I (Hz)')
    grid on
    
    subplot(3,3,6)
    % histogram(fic_t(:,rand_reg),'edgecolor','none');hold on
    h=histfit(fic_t(:,rand_reg),[],'loglogistic');hold on
    h3=histfit(fic_t(:,rand_reg),[],'normal');hold on
    set(h3(1),'facecolor','none','edgecolor','none')
    set(h(1),'facecolor',get(p3,'color'),'edgecolor','none')
    % set(h3(1),'edgecolor','none')
    set(h(1),'edgecolor','none','facealpha',0.8)
    set(h(2),'color',[0 0 0.4],'linewidth',2)
    set(h3(2),'color',[0.6 0 0],'linewidth',2)
    legend([h(2),h3(2)],'log-logistic fit','normal fit')
    xlabel('J_n')
    grid on
    
    subplot(3,3,9)
    p1=plot(stren,mean(fic_t),'.','markersize',10);hold on
    lsline
    p2=plot(stren,mean_fic_dist,'.','markersize',10);hold on
    lsline
    
    legend([p1,p2],'log-logistic fit','normal fit','location','northwest')
    grid on
    xlabel('Strength')
    ylabel('Mean FIC')
    
%     print(gcf,'-dpng',[figfold,figfile,'.png'],'-r300')
%     print(gcf,'-dpdf',[figfold,figfile,'.pdf'],'-r300')
%     
%     close gcf
    
end
