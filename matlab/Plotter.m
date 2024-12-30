results_dyn = load('/network/iss/home/ivan.mindlin/dyn_fic_dmf/matlab/Results/dyn_fcd/results_GoodRange_2.mat');
results_dyn = results_dyn.BayesoptResults;
results_stat = load('/network/iss/home/ivan.mindlin/dyn_fic_dmf/matlab/Results/stat_fcd/results_GoodRange_2.mat');
results_stat = results_stat.BayesoptResults;

% Plotting the training curves
subplot(1,2,1);
plot(results_stat.ObjectiveMinimumTrace, '--b', 'LineWidth', 2);  % Blue dashed line
hold on;
plot(results_stat.EstimatedObjectiveMinimumTrace, '-g', 'LineWidth', 2);  % Green solid line

% Adding labels and title
xlabel('Iterations');
ylabel('1 - Correlation');
title('Static FIC - Fitting');

% Adding legend
legend('Objective', 'Estimated Objective');

% Display the plot
grid on;
hold off;
subplot(1,2,2);


plot(results_dyn.ObjectiveMinimumTrace, '--b', 'LineWidth', 2);  % Blue dashed line
hold on;
plot(results_dyn.EstimatedObjectiveMinimumTrace, '-g', 'LineWidth', 2);  % Green solid line

% Adding labels and title
xlabel('Iterations');
ylabel('1 - Correlation');
title('Dynamic FIC - Fitting');

% Adding legend
legend('Objective', 'Estimated Objective');

% Display the plot
grid on;
hold off;