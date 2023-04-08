% load('data_all.mat')
% load('../../Project_Files/distances.mat')


%%%------------------------------------
%%%------------------------------------
%%%             TASK 1
%%%------------------------------------
%%%------------------------------------

% disp("-----------------------");
% disp("Beginning task 1");
% 
% % disp("Calculate the distances (task 1)");
% % distances = calculate_distance(testv, trainv); % Matrix containing all distances
% % disp("Finished calculating the distances (task 1)");
% 
% disp("Classifying (1NN)");
% [conf_mat, w_data, w_labels, num_w, c_data, c_labels, num_c] = classify_1NN(distances, vec_size, num_test, testv, testlab, trainlab);
% disp("Finished classifying using the 1NN");
% 
% 
% % Plot one wrongly classified and one correctly classified
% plotting(w_data, w_labels, num_w, c_data, c_labels, num_c, col_size, row_size);
% 
% disp("Ending task 1");
% disp("-----------------------");

%%%------------------------------------
%%%------------------------------------
%%%         END OF TASK 1
%%%------------------------------------
%%%------------------------------------

%%%------------------------------------
%%%------------------------------------
%%%             TASK 2
%%%------------------------------------
%%%------------------------------------

disp("-----------------------");
disp("Beginning task 2");

M = 64;
[idx, C] = kmeans(trainv, M);

disp("Ending task 2");
disp("-----------------------");

%%%------------------------------------
%%%------------------------------------
%%%         END OF TASK 2
%%%------------------------------------
%%%------------------------------------

% Classifying the test-vectors given the distance matrix.
% Returns confusion-matrix, data, labels and amount for wrong and correct classification. 
function [cm, wd, wl, w, cd, cl, c] = classify_1NN(distances, vec_size, num_test, testv, testlab, trainlab)
    N = 10; % Number of classes, range is 0-9

    w = 0; % Amount of wrong classification
    c = 0; % Amount of correct classification

    cm = zeros(N, N); % Confusion-matrix

    wd = zeros(1, vec_size); % Vector containing data for wrong classification
    wl = zeros(1, 2); % Vector containing labels for wrong classification
    cd = zeros(1, vec_size); % Vector containing data for correct classification
    cl = zeros(1, 2); % Vector containing labels for correct classification
    % For the label-matrix: First is the true label, the second is the
    % predicted label for each value

    for i = 1:num_test
        [~, index] = min(distances(:, i)); % Finds index in training set with min. dist.
        pl = trainlab(index); % Predicted/Classified label
        tl = testlab(i); % True label

        cm(tl+1, pl+1) = cm(tl+1, pl+1) + 1; % Updating confusion matrix

        % Checks is the predicted label was correct or wrong
        % Respectively adds data and [true label, predicted label] to
        % matrices
        if pl ~= tl
            w = w + 1; % Updates number of wrong classification
            wd(w, :) = testv(i, :); % Adds data
            wl(w, :) = [tl, pl]; % Adds true / predicted labels
        elseif pl == tl
            c = c + 1; % Updates number of correct classification
            cd(c, :) = testv(i, :); % Adds data
            cl(c, :) = [tl, pl]; % Adds true / predicted labels
        end
    end
end

% Plotting randomly picked wrongly classified and correctly classified
function plotting(wrong_data, wrong_labels, num_wrong, correct_data, correct_labels, num_correct, col_size, row_size)
    randint_wrong = randi(num_wrong); % Index [1, # wrongly classified]
    image_wrong = wrong_data(randint_wrong, :);

    randint_correct = randi(num_correct); % Index [1, # correctly classified]
    image_correct = correct_data(randint_correct, :);

    image(transpose(reshape(image_wrong, col_size, row_size)));
    disp("For the wrongly classified: ");
    fprintf('True label: ');
    fprintf('%d', wrong_labels(randint_wrong, 1));
    fprintf(' Predicted label: ');
    fprintf('%d', wrong_labels(randint_wrong, 2));
    fprintf('\n');
    pause(5);
    image(transpose(reshape(image_correct, col_size, row_size)));
    disp("For the correctly classified: ");
    fprintf('True label: ');
    fprintf('%d', correct_labels(randint_correct, 1));
    fprintf(' Predicted label: ');
    fprintf('%d', correct_labels(randint_correct, 2));
    fprintf('\n');
end

% Calculates the distances given test-vector and templates
function distances = calculate_distance(test_set, templates)
    distances = dist(templates, transpose(test_set));
end
