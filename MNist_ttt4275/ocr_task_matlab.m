% disp("Loading data");
% load('data_all.mat')
% disp("Loading pre-saved distances");
% load('../../Project_Files/distances.mat')


%%%------------------------------------
%%%------------------------------------
%%%             TASK 1
%%%------------------------------------
%%%------------------------------------

% disp("-----------------------");
% disp("Beginning task 1");
% 
% % Task 1 variables
% 
% % Used in 1NN-classification
% % w_1 & c_1 : Number of wrong and correct classifications
% % cm_1 : Confusion-matrix
% % wd_1 & wl_1 : Array containing data and labels respectively for wrongly classified images
% % cd_1 & cl_1 : Same as the above, only with correctly classified images
% % The label matrices contain [True label, Predicted label]
% [w_1, c_1, cm_1, wd_1, wl_1, cd_1, cl_1] = deal(0, 0, zeros(10, 10), zeros(1, vec_size), zeros(1, 2), zeros(1, vec_size), zeros(1, 2));
% 
% % Same as the above, only that they are used in the 7NN-classification
% [w_1_7, c_1_7, cm_1_7, wd_1_7, wl_1_7, cd_1_7, cl_1_7] = deal(0, 0, zeros(10, 10), zeros(1, vec_size), zeros(1, 2), zeros(1, vec_size), zeros(1, 2));
% 
% size_bulk = 999; % 1000 - 1 (needs to be removed)
% for i = 1:(num_test/size_bulk)
%     % Splitting the testing-sets into 'bulks' of 1000 elements, calculating
%     % distances for each of these sets. Adding all togheter to a new matrix
% %     distances_sets = zeros(num_train, size_bulk+1); 
%     distances_set = distances(:, i:i+size_bulk);
%     testing_data = testv(i:i+size_bulk, :);
%     testing_labels = testlab(i:i+size_bulk, :);
%     [number_of_tests, ~] = size(testing_data);
%     
% %     disp("Calculating distances");
% %     distances_sets = calculate_distance(testing_data, trainv);
%     disp("Classifying");
%     % 1-NN classifier:
%     [cm_1, wd_1, wl_1, w_1, cd_1, cl_1, c_1] = classify_1NN(distances_set, number_of_tests, testing_data, testing_labels, trainlab, cm_1, wd_1, wl_1, w_1, cd_1, cl_1, c_1);
%     % 7-NN classifier:
%     [cm_1_7, wd_1_7, wl_1_7, w_1_7, cd_1_7, cl_1_7, c_1_7] = classify_kNN(distances_set, number_of_tests, testing_data, testing_labels, trainlab, cm_1_7, wd_1_7, wl_1_7, w_1_7, cd_1_7, cl_1_7, c_1_7, 7);
% end
% 
% error_rate_1 = w_1/num_test;
% error_rate_1_7 = w_1_7/num_test;
% 
% % Plot one wrongly classified and one correctly classified
% % plotting(w_data, w_labels, num_w, c_data, c_labels, num_c, col_size, row_size);
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

% disp("-----------------------");
% disp("Beginning task 2");
% 
% disp("Sorting & Clustering");
% 
% % Creating vectors to hold all training data of one class
% % [trainv0, trainv1, trainv2, trainv3, trainv4, trainv5, trainv6, trainv7, trainv8, trainv9] = sorting(trainv, trainlab, num_train, vec_size);
% 
% % Returns a 1x10 cell
% % Each element, f. ex. trainv_sorted{1} contains all the data with label 0 (digit 0)
% trainv_sorted = sorting(trainv, trainlab, num_train, vec_size);
% 
% M = 64;
% new_training_set = zeros(10*M, vec_size);
% 
% for i = 0:9
%    [~, Ci] = kmeans(trainv_sorted{i+1}, M);
%    new_training_set(i*M+1:(i+1)*M, :) = Ci;
% end
% 
% % Task 2 variables
% 
% % Used in 1NN-classification
% % w_2 & c_2 : Number of wrong and correct classifications
% % cm_2 : Confusion-matrix
% % wd_2 & wl_2 : Array containing data and labels respectively for wrongly classified images
% % cd_2 & cl_2 : Same as the above, only with correctly classified images
% % The label matrices contain [True label, Predicted label]
% [w_2, c_2, cm_2, wd_2, wl_2, cd_2, cl_2] = deal(0, 0, zeros(10, 10), zeros(1, vec_size), zeros(1, 2), zeros(1, vec_size), zeros(1, 2));
% 
% % 7-NN classifier:
% [w_2_7, c_2_7, cm_2_7, wd_2_7, wl_2_7, cd_2_7, cl_2_7] = deal(0, 0, zeros(10, 10), zeros(1, vec_size), zeros(1, 2), zeros(1, vec_size), zeros(1, 2));
% 
% size_bulk = 999; % 1000 - 1 (needs to be removed)
% for i = 1:(num_test/(size_bulk+1))
%     % Splitting the testing-sets into 'bulks' of 1000 elements, calculating
%     % distances for each of these sets. Adding all togheter to a new matrix
%     testing_data = testv(i:i+size_bulk, :);
%     testing_labels = testlab(i:i+size_bulk, :);
%     [number_of_tests, ~] = size(testing_data);
%     
%     training_labels = [0*ones(M, 1); 1*ones(M,1); 2*ones(M,1); 3*ones(M,1); 4*ones(M,1); 5*ones(M,1); 6*ones(M,1); 7*ones(M,1); 8*ones(M,1); 9*ones(M,1)];
%     
%     disp("Calculating distances");
%     distances_clustering = calculate_distance(testing_data, new_training_set);
%     disp("Classifying");
%     
%     % 1-NN classifier
%     [cm_2, wd_2, wl_2, w_2, cd_2, cl_2, c_2] = classify_1NN(distances_clustering, number_of_tests, testing_data, testing_labels, training_labels, cm_2, wd_2, wl_2, w_2, cd_2, cl_2, c_2);
%     % 7-NN classifier
%     [cm_2_7, wd_2_7, wl_2_7, w_2_7, cd_2_7, cl_2_7, c_2_7] = classify_kNN(distances_clustering, number_of_tests, testing_data, testing_labels, training_labels, cm_2_7, wd_2_7, wl_2_7, w_2_7, cd_2_7, cl_2_7, c_2_7, 7);
% end
% 
% error_rate_2 = w_2/num_test;
% error_rate_2_7 = w_2_7/num_test;
% 
% disp("Ending task 2");
% disp("-----------------------");

%%%------------------------------------
%%%------------------------------------
%%%         END OF TASK 2
%%%------------------------------------
%%%------------------------------------

% plotting(wd_2(1, :), col_size, row_size, wl_2(1, 1), wl_2(1, 2));

% Classifying the test-vectors given the distance matrix using 1NN.
% Returns confusion-matrix, data, labels and amount for wrong and correct classification. 
function [cm, wd, wl, w, cd, cl, c] = classify_kNN(distances_set, num_test, test_data, test_labels, training_labels, cm, wd, wl, w, cd, cl, c, k)
    for l = 1:num_test
        [~, indices] = sort(distances_set(:, l));
        
        min_labels = zeros(k, 1);
        for y = 1:k
            min_labels(y) = training_labels(indices(y));
        end

        pl = mode(min_labels); % Finding the predicted label (Most frequent of the 7 labels)
        tl = test_labels(l); % Finding the true label
        
        cm(tl+1, pl+1) = cm(tl+1, pl+1) + 1; % Updating confusion matrix

        % Checks is the predicted label was correct or wrong
        % Respectively adds data and [true label, predicted label] to matrices
        if pl ~= tl
            w = w + 1; % Updates number of wrong classification
            wd(w, :) = test_data(l, :); % Adds data
            wl(w, :) = [tl, pl]; % Adds true / predicted labels
        elseif pl == tl
            c = c + 1; % Updates number of correct classification
            cd(c, :) = test_data(l, :); % Adds data
            cl(c, :) = [tl, pl]; % Adds true / predicted labels
        end
    end
end

% Classifying the test-vectors given the distance matrix using 1NN.
% Returns confusion-matrix, data, labels and amount for wrong and correct classification. 
function [cm, wd, wl, w, cd, cl, c] = classify_1NN(distances_set, num_test, test_data, test_labels, training_labels, cm, wd, wl, w, cd, cl, c)
    for i = 1:num_test
        [~, index] = min(distances_set(:, i)); % Finds index in training set with min. dist.
        pl = training_labels(index); % Predicted/Classified label
        tl = test_labels(i); % True label

        cm(tl+1, pl+1) = cm(tl+1, pl+1) + 1; % Updating confusion matrix

        % Checks is the predicted label was correct or wrong
        % Respectively adds data and [true label, predicted label] to
        % matrices
        if pl ~= tl
            w = w + 1; % Updates number of wrong classification
            wd(w, :) = test_data(i, :); % Adds data
            wl(w, :) = [tl, pl]; % Adds true / predicted labels
        elseif pl == tl
            c = c + 1; % Updates number of correct classification
            cd(c, :) = test_data(i, :); % Adds data
            cl(c, :) = [tl, pl]; % Adds true / predicted labels
        end
    end
end

% Plotting randomly picked wrongly classified and correctly classified
function plotting(image_data, col_size, row_size, tl, pl)
    image(transpose(reshape(image_data, col_size, row_size)));
    title("True label: " + num2str(tl) + " - Classified label: " + num2str(pl));
end

% Calculates the distances given test-vector and templates
function distances_return = calculate_distance(test_set, templates)
    distances_return = dist(templates, transpose(test_set));
end

function s = sorting(training_data, training_label, num_train, vec_size)
    s = {zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size)};
    for j = 1:num_train
        s{training_label(j)+1}(end+1, :) = training_data(j, :);
        disp(j);
    end
end
