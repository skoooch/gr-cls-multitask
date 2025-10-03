
selected_object = 'ob1'; 
selected_channel = 1; 

struct_names = {'plant', 'figurine', 'pen', 'chair', 'lamp'};
plot_colors = lines(length(struct_names)); 

figure;
hold on;

for i = 1:length(struct_names)
    struct_data = eval(struct_names{i});
    data = struct_data.(selected_object); 
    
    avg_data = squeeze(mean(data(:, selected_channel, :), 1));

    n_timepoints = size(data, 3);
    time = linspace(-0.1, 0.5, n_timepoints); 

    plot(time, avg_data, 'Color', plot_colors(i, :), 'LineWidth', 2, 'DisplayName', struct_names{i});
end

xlabel('Time (s)');
ylabel('Averaged Channel Data');
title(['Averaged Epochs Across Objects for ', selected_object, ', Channel ', num2str(selected_channel-1)]);
legend('Location', 'northeast'); 
grid on;
hold off;