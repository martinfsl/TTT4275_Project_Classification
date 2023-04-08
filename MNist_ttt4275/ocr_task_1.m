% Saves all the data-points from the first image
% x = testv(1, :);

%distance = dist(trainv, transpose(testv(1, :)));

% Calculates the distances
%distances = dist(trainv, transpose(testv));

% Predicted label

N = 10; % Number of classes, range is 0-9

wrong = 0;

confusion_matrix = zeros(N, N);

for i = 1:num_test
    [min_value, index] = min(distances(:, i));
    pl = trainlab(index); % Predicted/Classified label
    tl = testlab(i); % True label
    
    confusion_matrix(tl+1, pl+1) = confusion_matrix(tl+1, pl+1) + 1; % Updating confusion matrix
    
    if pl ~= tl
        wrong = wrong + 1;
    end
end

confusion_matrix(1, 5) = confusion_matrix(1,5) + 1;

