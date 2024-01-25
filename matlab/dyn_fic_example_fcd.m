%% Ejemplo FCd
% Fetch default parameters
load SC_and_5ht2a_receptors.mat
C = 0.2.*sc90./max(sc90(:));
params = dyn_fic_DefaultParams('C',C);
N=length(params.C);
% Setting model parameters
params.G = 2;
params.seed = 1;
% FIC parameters
% params.J = 0.75*params.G*sum(params.C, 1)' + 1; % FIC initial value
params.J = 0.75*params.G*sum(params.C, 1)' + 1; % FIC initial value
params.obj_rate = 3.44; % FIC objective rate

% ---------------------- Parameters for edge/slow oscillations with G=4
params.lrj = 30; % FIC learning Rate
params.taoj = 20000; % FIC decay

% ---------------------- Parameters for high frequency oscillations with G=4
% params.lrj = 10; % FIC learning Rate
% params.taoj = 5000; % FIC decay

% ---------------------- Parameters for noisy with G=4
% params.lrj = 30; % FIC learning Rate
% params.taoj = 20000; % FIC decay


% Running Simulation with Dynamic FIC
brunout = 5; % seconds
nb_steps = 100000;


[rates, rates_inh, fic_t] = dyn_fic_DMF(params, nb_steps,'ratefic');

%%


bold = rand(N, size(rates,2));
%
rates = rates(:,ceil(brunout*1000):end);
rates_inh = rates_inh(:,ceil(brunout*1000):end);
fic_t = fic_t(:,ceil(brunout*1000):end);
% Computing FCS
rates_fc = corrcoef(rates');
rates_inh_fc = corrcoef(rates_inh');
%bold_fc = corrcoef(bold');
%
% Computing FCd
T = length(bold);
isubfcd = find(tril(ones(N),-1));
flp = 0.01;
fhp = 0.1;
wsize = 30;
overlap = 29;
win_start = 0:wsize-overlap:T-wsize-1;
nwins = length(win_start);
nints = length(isubfcd);
% Filtering bold
filt_bold = bold';%filter_bold(bold',flp,fhp,params.TR);
bold_fc = corrcoef(filt_bold);
time_fc= compute_fcd(filt_bold,wsize,overlap,isubfcd);


fcd = corrcoef(time_fc');
mean_fic_dist = zeros(N,1);
for n=1:N
    data = fic_t(:,n);
%     data(data<=0) = eps;
%     pd = fitdist(data,'loglogistic');
%     mean_fic_dist(n) = mean(pd);
    mean_fic_dist(n) = mean(data);
end

% Plotting
rand_reg = randi(N);
% sel_t = 2000:50000;
sel_t = 1:size(rates,2);
sel_t_bold = 1:size(bold,2);
%%
figure('units','normalized','outerposition',[0 0 1 1],'PaperOrientation','landscape','visible','on');
subplot(3,5,1:5)

% Plot the main plot with the left y-axis
subplot(3, 5, 1:5)
yyaxis left
plot(sel_t / 1000, mean(rates_inh(:, sel_t)), 'b'); hold on
plot(sel_t / 1000, mean(rates(:, sel_t)), 'r');
xlabel('Time (s)')
ylabel('E Firing Rates (Hz)')

% Create a second y-axis for fic_t on the right side
yyaxis right
plot(sel_t / 1000, mean(fic_t(:, sel_t)), 'color', [0 0.5 0]); % Customizing color for fic_t
ylabel('FIC Y-axis') % Replace 'FIC Y-axis Label' with the appropriate label for fic_t
hold off

%
subplot(3,5,6:10)


%plot(sel_t_bold,filt_bold(sel_t_bold,:));
%set(gca,'xticklabel',(sel_t_bold-1).*params.TR);
%xlim([0 sel_t_bold(end)])
%xlabel('Time (s)')
%ylabel('BOLD(Hz)')

subplot(3,5,11)
bar(mean(rates,2),'edgecolor','none');hold on
p2=plot([0 N+1],[params.obj_rate params.obj_rate],'r--');
ylabel('E Firing Rate (Hz)')
xlabel('Regions')
legend(p2,'Objective Rate','location','southeast')

subplot(3,5,12)
imagesc(rates_fc-eye(N))
axis square
title('FC Rates')
colorbar

subplot(3,5,13)
%imagesc(bold_fc-eye(N))
%axis square
%title('FC BOLD')
%colorbar


subplot(3,5,14)
%plot(f(freqIndices), PSD(freqIndices));
%ylim([min(PSD(freqIndices)),max(PSD(freqIndices))])
%xlabel('Frequencies')
%ylabel('Power')

subplot(3,5,15)
%imagesc(fcd)
%axis square
%title('FCD')
%colorbar


