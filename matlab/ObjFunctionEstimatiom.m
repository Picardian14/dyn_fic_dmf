close all; clear; clc;

num_points_y = 100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMIC FCD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plotScenario( ...
    './Results/dyn_fcd/HCP_Awake_sameranges.mat', ...
    './Results/dyn_fcd/results_HCP_Deep_sameranges_Explor.mat', ...
    'lr', logspace(0, 3, num_points_y), 'Learning Rate', 'log', ...
    'Results/dyn_fcd/results_awake_dyn_fcd.mat', ...
    'Results/dyn_fcd/results_deep_dyn_fcd.mat', ...
    'Dynamic FCD fit', ...
    true ... % Plot contours
);
sgtitle("Dynamic FCD fit"); % Add a title for the whole figure
% Save figure
% savefig('plots/Awake-vs-Deep-FCD_Dynamic.fig');
% saveas(gcf, 'plots/Awake-vs-Deep-FCD_Dynamic.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMIC FC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plotScenario( ...
    './Results/dyn_fc/HCP_Awake_sameranges.mat', ...
    './Results/dyn_fc/HCP_Deep_sameranges.mat', ...
    'lr', logspace(0, 3, num_points_y), 'Learning Rate', 'log', ...
    'Results/dyn_fc/results_awake_dyn_fc.mat', ...
    'Results/dyn_fc/results_deep_dyn_fc.mat', ...
    'Dynamic FC fit', ...
    false ... % Do not plot contours
);
sgtitle("Dynamic FC fit"); % Add a title for the whole figure
% Save figure
% savefig('plots/Awake-vs-Deep-FC_Dynamic.fig');
% saveas(gcf, 'plots/Awake-vs-Deep-FC_Dynamic.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STATIC FCD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plotScenario( ...
    './Results/stat_fcd/HCP_Awake_sameranges.mat', ...
    './Results/stat_fcd/HCP_Deep_sameranges.mat', ...
    'alpha', linspace(0.65, 0.85, num_points_y), 'Alpha', 'linear', ...
    'Results/stat_fcd/results_awake_stat_fcd.mat', ...
    'Results/stat_fcd/results_deep_stat_fcd.mat', ...
    'Static FCD fit', ...
    true ... % Plot contours if applicable
);
sgtitle("Static FCD fit"); % Add a title for the whole figure
% Save figure
% savefig('plots/Awake-vs-Deep-FCD_Static.fig');
% saveas(gcf, 'plots/Awake-vs-Deep-FCD_Static.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STATIC FC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plotScenario( ...
    './Results/stat_fc/HCP_Awake_sameranges.mat', ...
    './Results/stat_fc/HCP_Deep_sameranges.mat', ...
    'alpha', linspace(0.65, 0.85, num_points_y), 'Alpha', 'linear', ...
    'Results/stat_fc/results_awake_stat_fc.mat', ...
    'Results/stat_fc/results_deep_stat_fc.mat', ...
    'Static FC fit', ...
    false ... % Do not plot contours
);
sgtitle("Static FC fit"); % Add a title for the whole figure
% Save figure
% savefig('plots/Awake-vs-Deep-FC_Static.fig');
% saveas(gcf, 'plots/Awake-vs-Deep-FC_Static.png');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper Function: plotScenario
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotScenario(resultFileAwake, resultFileDeep,yVarName, ySpace, yLabelName, yScaleType, savePathAwake, savePathDeep, mainTitle, plotContour)
    % Load Awake data
    % Common parameters
    num_points_y = 100;
    num_points_x = 60;
    G_space = linspace(0.1, 16, num_points_x);

    boAwakeData = load(resultFileAwake);
    if isfield(boAwakeData, 'results')
        boAwake = boAwakeData.results;
    elseif isfield(boAwakeData, 'BayesoptResults')
        boAwake = boAwakeData.BayesoptResults;
    else
        error('Unexpected structure in %s', resultFileAwake);
    end
    
    XatminAwake = boAwake.XAtMinObjective;
    XatminEstimatedAwake = boAwake.XAtMinEstimatedObjective;

    % Create grid for Awake
    [Y, G] = meshgrid(ySpace, G_space);
    G = G(:);
    Y = Y(:);
    grid_points = table(Y, G);
    grid_points.Properties.VariableNames = {yVarName, 'G'};

    % Predict objective for Awake
    [o, ~] = predictObjective(boAwake, grid_points);
    funcEvalsAwake = reshape(o, num_points_x, num_points_y);

    % Calculate the threshold for the top 5% minimum values
    sortedValues = sort(funcEvalsAwake(:));
    top5PercentThreshold = sortedValues(round(0.05 * length(sortedValues)));
    % Create a binary mask where values are below this threshold
    minRegionMask = funcEvalsAwake <= top5PercentThreshold;

    % Plot Awake
    subplot(1, 2, 1);
    imagesc(G_space, ySpace, funcEvalsAwake');
    set(gca, 'YDir', 'normal');
    if strcmpi(yScaleType, 'log')
        set(gca, 'YScale', 'log');
    end
    title("Awake");
    xlabel("G");
    ylabel(yLabelName);
    hold on;
    
    if plotContour
        % Overlay the contour for the top 5% minimum region
        contour(G_space, ySpace, minRegionMask', [1 1], 'LineColor', 'w', 'LineWidth', 1.5);
    end
    
    % Plot minimum points
    plot(XatminAwake.G, XatminAwake.(yVarName), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    plot(XatminEstimatedAwake.G, XatminEstimatedAwake.(yVarName), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
    legend('Xatmin', 'XatminEstimated');
    hold off;

    % Save Awake results
    minEstimatedG_Awake = XatminEstimatedAwake.G;
    minEstimatedY_Awake = XatminEstimatedAwake.(yVarName);
    funcEvals_data_Awake = funcEvalsAwake;
    save(savePathAwake, 'minEstimatedY_Awake', 'minEstimatedG_Awake', 'funcEvals_data_Awake', 'ySpace', 'G_space','minRegionMask');

    % Load Deep data
    boDeepData = load(resultFileDeep);
    if isfield(boDeepData, 'results')
        boDeep = boDeepData.results;
    elseif isfield(boDeepData, 'BayesoptResults')
        boDeep = boDeepData.BayesoptResults;
    else
        error('Unexpected structure in %s', resultFileDeep);
    end
    
    XatminDeep = boDeep.XAtMinObjective;
    XatminEstimatedDeep = boDeep.XAtMinEstimatedObjective;

    % Predict objective for Deep
    [o, ~] = predictObjective(boDeep, grid_points);
    funcEvalsDeep = reshape(o, num_points_x, num_points_y);

    % Calculate threshold and mask for Deep
    sortedValuesDeep = sort(funcEvalsDeep(:));
    top5PercentThresholdDeep = sortedValuesDeep(round(0.05 * length(sortedValuesDeep)));
    minRegionMaskDeep = funcEvalsDeep <= top5PercentThresholdDeep;

    % Plot Deep
    subplot(1, 2, 2);
    imagesc(G_space, ySpace, funcEvalsDeep');
    set(gca, 'YDir', 'normal');
    if strcmpi(yScaleType, 'log')
        set(gca, 'YScale', 'log');
    end
    title("Deep");
    xlabel("G");
    ylabel(yLabelName);
    hold on;
    
    if plotContour
        % Overlay the contour for the top 5% minimum region
        contour(G_space, ySpace, minRegionMaskDeep', [1 1], 'LineColor', 'w', 'LineWidth', 1.5);
    end
    
    % Plot minimum points
    plot(XatminDeep.G, XatminDeep.(yVarName), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    plot(XatminEstimatedDeep.G, XatminEstimatedDeep.(yVarName), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
    legend('Xatmin', 'XatminEstimated');
    hold off;

    % Save Deep results
    minEstimatedG_Deep = XatminEstimatedDeep.G;
    minEstimatedY_Deep = XatminEstimatedDeep.(yVarName);
    funcEvals_data_Deep = funcEvalsDeep;
    save(savePathDeep, 'minEstimatedY_Deep', 'minEstimatedG_Deep', 'funcEvals_data_Deep', 'ySpace', 'G_space','minRegionMaskDeep');
end
