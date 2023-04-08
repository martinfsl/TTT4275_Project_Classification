% Saves all the data-points from the first image
% x = testv(1, :);

% distance = dist(trainv, transpose(testv(1, :))); % One test-vector

% % Calculates the distances
% distances = dist(trainv, transpose(testv));

% % Classifying the test-vectors given the distance matrix
% N = 10; % Number of classes, range is 0-9
% 
% wrong = 0;
% correct = 0;
% 
% confusion_matrix = zeros(N, N);
% 
% wrongly_classified = zeros(1, vec_size);
% wrongly_labels = zeros(1, 2); % First is the true label, the second is the predicted label
% correctly_classified = zeros(1, vec_size);
% correctly_labels = zeros(1, 2);
% 
% for i = 1:num_test
%     [min_value, index] = min(distances(:, i));
%     pl = trainlab(index); % Predicted/Classified label
%     tl = testlab(i); % True label
%     
%     confusion_matrix(tl+1, pl+1) = confusion_matrix(tl+1, pl+1) + 1; % Updating confusion matrix
%     
%     if pl ~= tl
%         wrong = wrong + 1;
%         wrongly_classified(wrong, :) = testv(i, :);
%         wrongly_labels(wrong, :) = [tl, pl];
%     elseif pl == tl
%         correct = correct + 1;
%         correctly_classified(correct, :) = testv(i, :);
%         correctly_labels(correct, :) = [tl, pl];
%     end
% end
% 
% confusion_matrix(1, 5) = confusion_matrix(1,5) + 1;

% % Plotting randomly picked wrongly classified and correctly classified
% randint_wrong = randi(wrong); % Index [1, # wrongly classified]
% image_wrong = wrongly_classified(randint_wrong, :);
% 
% randint_correct = randi(correct); % Index [1, # correctly classified]
% image_correct = correctly_classified(randint_correct, :);
% 
% image(transpose(reshape(image_wrong, col_size, row_size)));
% disp("For the wrongly classified: ");
% disp(wrongly_labels(randint_wrong, :));
% disp("[True Classified]");
% pause(5);
% image(transpose(reshape(image_correct, col_size, row_size)));
% disp("For the correctly classified: ");
% disp(correctly_labels(randint_correct, :));
% disp("[True Classified]");
