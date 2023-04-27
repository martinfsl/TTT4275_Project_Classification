plotting_clustered_data(new_training_set((1+9*64):((1+9)*64), :), 9);

function plotting_clustered_data(array, digit)

%     for i = 1:64
%         disp(i);
%     end
    
    figure('Position', [100 100 500 500]);
    
    for i = 1:8
        for j = 0:7
%           disp((i-1)*8+j+1);
%           subplot('Position', [(0.025+0.05*(i-1)) (0.925-0.055*j) 0.05 0.05]);
%             subplot('Position', [(0.210+0.075*(i-1)) (0.850-0.065*j) 0.05 0.05]);
            subplot('Position', [(0.115+0.1*(i-1)) (0.800-0.1*j) 0.075 0.075]);
            imshow(transpose(reshape(array((i-1)*8+j+1, :), 28, 28)), []);
        end
    end
    
    sgtitle("The clustered templates for the digit " + digit);
    saveas(gcf, 'clusters_plots/clusters_9.png');
    
%     subplot('Position', [0.025 0.925 0.05 0.05]);
%     imshow(transpose(reshape(array(1, :), 28, 28)), []);
%     subplot('Position', [0.025 0.87 0.05 0.05]);
%     imshow(transpose(reshape(array(2, :), 28, 28)), []);
%     subplot('Position', [0.075 0.925 0.05 0.05]);
%     imshow(transpose(reshape(array(3, :), 28, 28)), []);
    
end