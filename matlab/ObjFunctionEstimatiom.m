%% DYNAMIC
figure;
% Load and process the Awake data
bo = load('./Results/dyn_fcd/HCP_Awake_sameranges.mat'); 
boAwake = bo.results;
XatminAwake = boAwake.XAtMinObjective;
XatminEstimatedAwake = boAwake.XAtMinEstimatedObjective;
num_points_y = 90;
num_points_x = 100;
G_space = linspace(0.1,16, num_points_x);
lr_space = logspace(0, 3, num_points_y);

[lr, G] = meshgrid(lr_space, G_space);
G = G(:);
lr = lr(:);
grid_points = table(lr, G);
[o, s] = predictObjective(bo.results, grid_points);
funcEvalsAwake = reshape(o, num_points_x, num_points_y);

% Calculate the threshold for the top 10% minimum values
sortedValues = sort(funcEvalsAwake(:));
top5PercentThreshold = sortedValues(round(0.05 * length(sortedValues)));

% Create a binary mask where values are below this threshold
minRegionMask = funcEvalsAwake <= top5PercentThreshold;

% Plot Awake data
subplot(1, 2, 1);
imagesc(G_space, lr_space, funcEvalsAwake');
set(gca, 'YDir', 'normal', 'YScale', 'log');
title("Awake");
xlabel("G");
ylabel("Learning Rate");
hold on;

% Overlay the contour for the top 10% minimum region
contour(G_space, lr_space, minRegionMask', [1 1], 'LineColor', 'w', 'LineWidth', 1.5);

% Mark minimum points (optional)
plot(XatminAwake.G, XatminAwake.lr, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(XatminEstimatedAwake.G, XatminEstimatedAwake.lr, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Xatmin', 'XatminEstimated');
hold off;
G_dyn_fcd = XatminEstimatedAwake.G;
lr_dyn_fcd = XatminEstimatedAwake.lr;
funcEvals_dyn = funcEvalsAwake;
save("Results/dyn_fcd/results_awake_dyn_fcd.mat", "lr_dyn_fcd","G_dyn_fcd","funcEvals_dyn","lr_space","G_space")
% DEEP FCD
% Load and process the Deep data

bo = load('./Results/dyn_fcd/results_HCP_Deep_sameranges_Explor.mat'); 
boDeep = bo.BayesoptResults;
XatminDeep = boDeep.XAtMinObjective;
XatminEstimatedDeep = boDeep.XAtMinEstimatedObjective;

[lr, G] = meshgrid(lr_space, G_space);
G = G(:);
lr = lr(:);
grid_points = table(lr, G);
[o, s] = predictObjective(bo.BayesoptResults, grid_points);
funcEvalsDeep = reshape(o, num_points_x, num_points_y);

% Calculate the threshold for the top 10% minimum values
sortedValuesDeep = sort(funcEvalsDeep(:));
top5PercentThresholdDeep = sortedValuesDeep(round(0.05 * length(sortedValuesDeep)));

% Create a binary mask for values below the top 10% threshold
minRegionMaskDeep = funcEvalsDeep <= top5PercentThresholdDeep;

% Plot Deep data
subplot(1, 2, 2);
imagesc(G_space, lr_space, funcEvalsDeep');
set(gca, 'YDir', 'normal', 'YScale', 'log');
title("Deep");
xlabel("G");
ylabel("Learning Rate");
hold on;

% Overlay the contour for the top 10% minimum region
contour(G_space, lr_space, minRegionMaskDeep', [1 1], 'LineColor', 'w', 'LineWidth', 1.5);

% Plot minimum points
plot(XatminDeep.G, XatminDeep.lr, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(XatminEstimatedDeep.G, XatminEstimatedDeep.lr, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Xatmin', 'XatminEstimated');
hold off;
G_dyn_fcd = XatminEstimatedDeep.G;
lr_dyn_fcd = XatminEstimatedDeep.lr;
funcEvals_dyn = funcEvalsDeep;
save("Results/dyn_fcd/results_deep_dyn_fcd.mat", "lr_dyn_fcd","G_dyn_fcd","funcEvals_dyn","lr_space","G_space")

sgtitle("Dynamic FCD fit"); % Add a title for the whole figure
% Save figure
savefig('plots/Awake-vs-Deep-FCD_Dynamic.fig');
saveas(gcf, 'plots/Awake-vs-Deep-FCD_Dynamic.png')




%% DYNAMIC FC
figure;
bo = load('./Results/dyn_fc/HCP_Awake_sameranges.mat'); % Replace 'your_saved_bo_object.mat' with the actual filename
boAwake = bo.results;
num_points_y = 90;
num_points_x = 100;
G_space = linspace(0.1,16, num_points_x);
lr_space = logspace(0, 3, num_points_y);
[lr, G] = meshgrid(lr_space, G_space);
G = G(:);
lr = lr(:);
grid_points = table(lr, G);
[o,s] = predictObjective(bo.results, grid_points);
funcEvals = reshape(o,100,90);
%funcEvals = funcEvals/max(funcEvals);
subplot(1,2,1);imagesc(G_space,lr_space,funcEvals');
set(gca, 'ydir', 'normal');
set(gca, 'Yscale', 'log');
title("Awake")
xlabel("G")
ylabel("Learning Rate")
hold on;
% Plot minimum points
plot(boAwake.XAtMinObjective.G, boAwake.XAtMinObjective.lr, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(boAwake.XAtMinEstimatedObjective.G, boAwake.XAtMinEstimatedObjective.lr, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Xatmin', 'XatminEstimated');
hold off;

G_dyn_fc = XatminEstimatedAwake.G;
lr_dyn_fc = XatminEstimatedAwake.lr;
funcEvals_dyn = funcEvalsAwake;
save("Results/dyn_fc/results_awake_dyn_fc.mat", "lr_dyn_fc","G_dyn_fc","funcEvals_dyn","lr_space","G_space")


bo = load('./Results/dyn_fc/HCP_Deep_sameranges.mat'); % Replace 'your_saved_bo_object.mat' with the actual filename
boDeep = bo.results;
num_points_y = 90;
num_points_x = 100;
G_space = linspace(0.1,16, num_points_x);
lr_space = logspace(0, 3, num_points_y);
[lr, G] = meshgrid(lr_space, G_space);
G = G(:);
lr = lr(:);
grid_points = table(lr, G);
[o,s] = predictObjective(bo.results, grid_points);
funcEvals = reshape(o,100,90);
%funcEvals = funcEvals/max(funcEvals);
subplot(1,2,2);imagesc(G_space,lr_space,funcEvals');
set(gca, 'ydir', 'normal');
set(gca, 'Yscale', 'log');
title("Deep")
xlabel("G")
ylabel("Learning Rate")
hold on;
% Plot minimum points
plot(boDeep.XAtMinObjective.G, boDeep.XAtMinObjective.lr, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(boDeep.XAtMinEstimatedObjective.G, boDeep.XAtMinEstimatedObjective.lr, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Xatmin', 'XatminEstimated');
hold off;

G_dyn_fc = XatminEstimatedDeep.G;
lr_dyn_fc = XatminEstimatedDeep.lr;
funcEvals_dyn = funcEvalsDeep;
save("Results/dyn_fc/results_deep_dyn_fc.mat", "lr_dyn_fc","G_dyn_fc","funcEvals_dyn","lr_space","G_space")

sgtitle("Dynamic FC fit") % Add a title for the whole figure
savefig('plots/Awake-vs-Deep-FC_Dynamic.fig')
saveas(gcf, 'plots/Awake-vs-Deep-FC_Dynamic.png')
%%


%% STATIC

figure;
bo = load('./Results/stat_fcd/HCP_Awake_sameranges.mat'); % Replace 'your_saved_bo_object.mat' with the actual filename
boAwake = bo.results;
num_points_y = 90;
num_points_x = 100;
G_space = linspace(0.1,16, num_points_x);
alpha_space = linspace(0.65, 0.85, num_points_y);
[alpha, G] = meshgrid(alpha_space, G_space);
G = G(:);
alpha = alpha(:);
grid_points = table(alpha, G);
[o,s] = predictObjective(bo.results, grid_points);
funcEvals = reshape(o,100,90);
%funcEvals = funcEvals/max(funcEvals);
subplot(1,2,1);imagesc(G_space,alpha_space,funcEvals');set(gca, 'ydir', 'normal')
title("Awake")
xlabel("G")
ylabel("Alpha")
hold on;
plot(boAwake.XAtMinObjective.G, boAwake.XAtMinObjective.alpha, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(boAwake.XAtMinEstimatedObjective.G, boAwake.XAtMinEstimatedObjective.alpha, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Xatmin', 'XatminEstimated');
hold off;


G_stat_fcd = boAwake.XAtMinObjective.G;
alpha_stat_fcd = boAwake.XAtMinObjective.alpha;
funcEvals_stat = funcEvals;
save("Results/stat_fcd/results_awake_stat_fcd.mat", "alpha_stat_fcd","G_stat_fcd","funcEvals_stat","alpha_space","G_space")


%

bo = load('./Results/stat_fcd/HCP_Deep_sameranges.mat'); % Replace 'your_saved_bo_object.mat' with the actual filename
boDeep = bo.results;
num_points_y = 90;
num_points_x = 100;
G_space = linspace(0.1,16, num_points_x);
alpha_space = linspace(0.65, 0.85, num_points_y);
[alpha, G] = meshgrid(alpha_space, G_space);
G = G(:);
alpha = alpha(:);
grid_points = table(alpha, G);
[o,s] = predictObjective(bo.results, grid_points);
funcEvals = reshape(o,100,90);
%funcEvals = funcEvals/max(funcEvals);
subplot(1,2,2);imagesc(G_space,alpha_space,funcEvals');set(gca, 'ydir', 'normal')

title("Deep")
xlabel("G")
ylabel("Alpha")
hold on;
% Plot minimum points
plot(boDeep.XAtMinObjective.G, boDeep.XAtMinObjective.alpha, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(boDeep.XAtMinEstimatedObjective.G, boDeep.XAtMinEstimatedObjective.alpha, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Xatmin', 'XatminEstimated');
hold off;


G_stat_fcd = boDeep.XAtMinObjective.G;
alpha_stat_fcd = boDeep.XAtMinObjective.alpha;
funcEvals_stat = funcEvals;
save("Results/stat_fcd/results_deep_stat_fcd.mat", "alpha_stat_fcd","G_stat_fcd","funcEvals_stat","alpha_space","G_space")



sgtitle("Static FCD fit") % Add a title for the whole figure
savefig('plots/Awake-vs-Deep-FCD_Static.fig')
saveas(gcf, 'plots/Awake-vs-Deep-FCD_Static.png')
%
figure;
bo = load('./Results/stat_fc/HCP_Awake_sameranges.mat'); % Replace 'your_saved_bo_object.mat' with the actual filename
boAwake = bo.results;
num_points_y = 90;
num_points_x = 100;
G_space = linspace(0.1,16, num_points_x);
alpha_space = linspace(0.65, 0.85, num_points_y);
[alpha, G] = meshgrid(alpha_space, G_space);
G = G(:);
alpha = alpha(:);
grid_points = table(alpha, G);
[o,s] = predictObjective(bo.results, grid_points);
funcEvals = reshape(o,100,90);
%funcEvals = funcEvals/max(funcEvals);
subplot(1,2,1);imagesc(G_space,alpha_space,funcEvals');set(gca, 'ydir', 'normal')
title("FC fit Awake")
xlabel("G")
ylabel("Alpha")
hold on;
plot(boAwake.XAtMinObjective.G, boAwake.XAtMinObjective.alpha, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(boAwake.XAtMinEstimatedObjective.G, boAwake.XAtMinEstimatedObjective.alpha, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Xatmin', 'XatminEstimated');
hold off;
G_stat_fc = boAwake.XAtMinObjective.G;
alpha_stat_fc = boAwake.XAtMinObjective.alpha;
funcEvals_stat = funcEvals;
save("Results/stat_fc/results_awake_stat_fc.mat", "alpha_stat_fc","G_stat_fc","funcEvals_stat","alpha_space","G_space")


%

bo = load('./Results/stat_fc/HCP_Deep_sameranges.mat'); % Replace 'your_saved_bo_object.mat' with the actual filename
boDeep = bo.results;
num_points_y = 90;
num_points_x = 100;
G_space = linspace(0.1,16, num_points_x);
alpha_space = linspace(0.65, 0.85, num_points_y);
[alpha, G] = meshgrid(alpha_space, G_space);
G = G(:);
alpha = alpha(:);
grid_points = table(alpha, G);
[o,s] = predictObjective(bo.results, grid_points);
funcEvals = reshape(o,100,90);
%funcEvals = funcEvals/max(funcEvals);
subplot(1,2,2);imagesc(G_space,alpha_space,funcEvals');set(gca, 'ydir', 'normal')
title("FC fit Deep")
xlabel("G")
ylabel("Alpha")
hold on;
% Plot minimum points
plot(boDeep.XAtMinObjective.G, boDeep.XAtMinObjective.alpha, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(boDeep.XAtMinEstimatedObjective.G, boDeep.XAtMinEstimatedObjective.alpha, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Xatmin', 'XatminEstimated');
hold off;


G_stat_fc = boDeep.XAtMinObjective.G;
alpha_stat_fc = boDeep.XAtMinObjective.alpha;
funcEvals_stat = funcEvals;
save("Results/stat_fc/results_deep_stat_fc.mat", "alpha_stat_fc","G_stat_fc","funcEvals_stat","alpha_space","G_space")


sgtitle("Static FC fit") % Add a title for the whole figure
savefig('plots/Awake-vs-Deep-FC_Static.fig')
saveas(gcf, 'plots/Awake-vs-Deep-FC_Static.png')
%%

