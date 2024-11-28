%% COMPARING DEEP AND AWAKE CONDITIONS MANUALLY

[minValDeep, minIdxDeep] = min(funcEvalsDeep(:));
[minGIdxDeep, minLrIdxDeep] = ind2sub(size(funcEvalsDeep), minIdxDeep);
minGDeep = G_space(minGIdxDeep);
minLrDeep = lr_space(minLrIdxDeep);

% For Awake condition
[minValAwake, minIdxAwake] = min(funcEvalsAwake(:));
[minGIdxAwake, minLrIdxAwake] = ind2sub(size(funcEvalsAwake), minIdxAwake);
minGAwake = G_space(minGIdxAwake);
minLrAwake = lr_space(minLrIdxAwake);


figure;
% Plot for Deep condition
subplot(1,2,1);
contourf(G_space, lr_space, funcEvalsDeep', 20);
hold on;
plot(minGDeep, minLrDeep, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
title('Deep Condition');
xlabel('G');
ylabel('Learning Rate');
set(gca, 'YDir', 'normal');

% Plot for Awake condition
subplot(1,2,2);
contourf(G_space, lr_space, funcEvalsAwake', 20);
hold on;
plot(minGAwake, minLrAwake, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
title('Awake Condition');
xlabel('G');
ylabel('Learning Rate');
set(gca, 'YDir', 'normal');

figure;
contourf(G_space, lr_space, funcEvalsDeep', 20);
hold on;
contour(G_space, lr_space, funcEvalsAwake', 20, 'LineStyle', '--');
plot(minGDeep, minLrDeep, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(minGAwake, minLrAwake, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
legend('Deep', 'Awake', 'Min Deep', 'Min Awake');
xlabel('G');
ylabel('Learning Rate');
title('Comparison of Minima between Deep and Awake Conditions');
set(gca, 'YDir', 'normal');


% Compute difference
funcEvalsDiff = funcEvalsDeep - funcEvalsAwake;

% Plot the difference
figure;
contourf(G_space, lr_space, funcEvalsDiff', 20);
colorbar;
title('Difference between Deep and Awake Conditions');
xlabel('G');
ylabel('Learning Rate');
set(gca, 'YDir', 'normal');


% Assuming you have multiple runs or samples to compute standard deviations
% For illustration, let's compute the standard deviation (replace with actual data)

% Example standard deviations (replace with actual computations)
stdDeep = std(funcEvalsDeep(:));
stdAwake = std(funcEvalsAwake(:));

% Compute the difference in minima
deltaMin = minValDeep - minValAwake;

% Compute the standard error
standardError = sqrt(stdDeep^2 + stdAwake^2);

% Compute the Z-score
zScore = deltaMin / standardError;

% Compute the p-value
pValue = 2 * (1 - normcdf(abs(zScore)));

fprintf('Difference in minima: %f\n', deltaMin);
fprintf('Standard Error: %f\n', standardError);
fprintf('Z-score: %f\n', zScore);
fprintf('P-value: %f\n', pValue);

% Define thresholds (e.g., within 5% of the minimum value)
thresholdDeep = minValDeep * 1.05;
thresholdAwake = minValAwake * 1.05;

% Identify regions
regionDeep = funcEvalsDeep <= thresholdDeep;
regionAwake = funcEvalsAwake <= thresholdAwake;

% Plot regions
figure;
subplot(1,2,1);
imagesc(G_space, lr_space, regionDeep');
set(gca, 'YDir', 'normal');
title('Deep Condition Minima Region');
xlabel('G');
ylabel('Learning Rate');

subplot(1,2,2);
imagesc(G_space, lr_space, regionAwake');
set(gca, 'YDir', 'normal');
title('Awake Condition Minima Region');
xlabel('G');
ylabel('Learning Rate');

