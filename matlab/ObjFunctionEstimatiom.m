bo = load('./Results/dyn_fcd/results_GoodRange_2.mat'); % Replace 'your_saved_bo_object.mat' with the actual filename
num_points_y = 90;
num_points_x = 100;
G_space = linspace(1,16, num_points_x);
lr_space = logspace(0, 3, num_points_y);
[lr, G] = meshgrid(lr_space, G_space);
G = G(:);
lr = lr(:);
grid_points = table(lr, G);
[o,s] = predictObjective(bo.BayesoptResults, grid_points);


funcEvals = reshape(o,100,90);
%funcEvals = funcEvals/max(funcEvals);
subplot(1,2,1);imagesc(G_space,lr_space,funcEvals');set(gca, 'ydir', 'normal')
%%
bo_stat = load('./Results/stat_fcd/results_GoodRange_2.mat'); % Replace 'your_saved_bo_stat_object.mat' with the actual filename
num_points_y_stat = 90;
num_points_x_stat = 100;
G_space = linspace(1,5, num_points_x_stat);
alpha_space = linspace(0.65, 0.85, num_points_y_stat);
[alpha, G] = meshgrid(alpha_space, G_space);
G = G(:);
alpha = alpha(:);
grid_points = table(alpha, G);
[o_stat,s_stat] = predictObjective(bo_stat.BayesoptResults, grid_points);


funcEvals_stat = reshape(o_stat,100,90);


subplot(1,2,2);imagesc(G_space,alpha_space,funcEvals_stat');set(gca, 'ydir', 'normal')

% Show colorbar
colorbar;
%%
bo = load('./Results/dyn_fc/Above1GandLRrange.mat'); % Replace 'your_saved_bo_object.mat' with the actual filename
num_points_y = 90;
num_points_x = 100;
G_space = linspace(1,16, num_points_x);
lr_space = logspace(0, 3, num_points_y);
[lr, G] = meshgrid(lr_space, G_space);
G = G(:);
lr = lr(:);
grid_points = table(lr, G);
[o,s] = predictObjective(bo.results, grid_points);


funcEvals = reshape(o,100,90);
%funcEvals = funcEvals/max(funcEvals);
subplot(1,2,1);imagesc(G_space,lr_space,funcEvals');set(gca, 'ydir', 'normal')
%%
bo_stat = load('./Results/stat_fc/results_BigGRangeShortAlpha.mat'); % Replace 'your_saved_bo_stat_object.mat' with the actual filename
num_points_y_stat = 90;
num_points_x_stat = 100;
G_space = linspace(1,5, num_points_x_stat);
alpha_space = linspace(0.65, 0.85, num_points_y_stat);
[alpha, G] = meshgrid(alpha_space, G_space);
G = G(:);
alpha = alpha(:);
grid_points = table(alpha, G);
[o_stat,s_stat] = predictObjective(bo_stat.BayesoptResults, grid_points);


funcEvals_stat = reshape(o_stat,100,90);


subplot(1,2,2);imagesc(G_space,alpha_space,funcEvals_stat');set(gca, 'ydir', 'normal')

% Show colorbar
colorbar;