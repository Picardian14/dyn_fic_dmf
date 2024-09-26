% For every seed iterate and load the bo object and plot the objective function 
num_points_y = 90;
num_points_x = 100;
all_funcEvals_rescaled = zeros(30,num_points_x,num_points_y);
decay_space = logspace(2,6, num_points_x);
lr_space = logspace(-1, 3, num_points_y);
[lr, decay] = meshgrid(lr_space, decay_space);
decay = decay(:);
lr = lr(:);
grid_points = table(lr, decay);
for s=1:30
    filename = sprintf('./Results/Low_G_range/seed_%d.mat',s);
    bo = load(filename).results; % Replace 'your_saved_bo_object.mat' with the actual filename
    [o,~] = predictObjective(bo, grid_points);
    funcEvals = reshape(o,100,90);
    min_val = min(funcEvals(:));
    max_val = max(funcEvals(:));
    funcEvals_rescaled = 200 * (funcEvals - min_val) / (max_val - min_val) - 100;
    min_values = min(funcEvals_rescaled, [], 1);  % Min across rows (along decay axis)
    % Perform linear fit to the minimum values
    %coeffs = polyfit(log10(lr_space), min_values, 1);  % Linear fit in log10 space
    all_funcEvals_rescaled(s,:,:) = funcEvals_rescaled;
end

funcEvals = squeeze(mean(all_funcEvals_rescaled,1));
imagesc(decay_space,lr_space,abs(funcEvals));set(gca, 'ydir', 'normal')



