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
% % Initializing task 1 variables
% 
% % Used in 1NN-classification
% % w_1 & c_1 : Number of wrong and correct classifications
% % cm_1 : Confusion-matrix
% % wd_1 & wl_1 : Array containing data and labels respectively for wrongly classified images
% % cd_1 & cl_1 : Same as the above, only with correctly classified images
% % tp_lab_1 : An array that holds the true and predicted labels, used for plotting
% % The label matrices contain [True label, Predicted label]
% % Using deal to get all intialization on one line
% [w_1, c_1, cm_1, wd_1, wl_1, cd_1, cl_1, tp_lab_1] = deal(0, 0, zeros(10, 10), zeros(1, vec_size), zeros(1, 2), zeros(1, vec_size), zeros(1, 2), zeros(0, 2));
% 
% % Same as the above, only that they are used in the 7NN-classification
% [w_1_7, c_1_7, cm_1_7, wd_1_7, wl_1_7, cd_1_7, cl_1_7, tp_lab_1_7] = deal(0, 0, zeros(10, 10), zeros(1, vec_size), zeros(1, 2), zeros(1, vec_size), zeros(1, 2), zeros(0, 2));
% 
% disp('Calculating distances and classifying');
% size_bulk = 999; % 1000 - 1 (needs to be removed to account for indexing)
% for i = 1:(num_test/size_bulk)
%     % Splitting the testing-sets into 'bulks' of 1000 elements
%     testing_data = testv(i:i+size_bulk, :);
%     testing_labels = testlab(i:i+size_bulk, :);
%     [number_of_tests, ~] = size(testing_data);
%     % Calculating distances for the bulk of testing data
%     distances_set = calculate_distance(testing_data, trainv);
%     
%     % 1-NN classifier:
%     [cm_1, wd_1, wl_1, w_1, cd_1, cl_1, c_1, tp_lab_1] = classify_kNN(distances_set, number_of_tests, testing_data, testing_labels, trainlab, cm_1, wd_1, wl_1, w_1, cd_1, cl_1, c_1, 1, tp_lab_1);
%     % 7-NN classifier:
%     [cm_1_7, wd_1_7, wl_1_7, w_1_7, cd_1_7, cl_1_7, c_1_7, tp_lab_1_7] = classify_kNN(distances_set, number_of_tests, testing_data, testing_labels, trainlab, cm_1_7, wd_1_7, wl_1_7, w_1_7, cd_1_7, cl_1_7, c_1_7, 7, tp_lab_1_7);
% end
% 
% error_rate_1 = w_1/num_test;
% disp("Error-rate for the 1NN-classifier for the unclustered data: " + error_rate_1);
% error_rate_1_7 = w_1_7/num_test;
% disp("Error-rate for the 7NN-classifier for the unclustered data: " + error_rate_1_7);

% % Plotting the confusion matrices
% plot_confusion_matrix(tp_lab_1, "Confusion matrix for the unclustered data using the 1NN-classifier");
% pause(5);
% plot_confusion_matrix(tp_lab_1_7, "Confusion matrix for the unclustered data using the 7NN-classifier");
% pause(5);
% 
% % 1NN
% % Picking three random correctly classified digits to plot
% [ri_1_c1, ri_1_c2, ri_1_c3] = deal(randi(length(cl_1), 1), randi(length(cl_1), 1), randi(length(cl_1), 1));
% labels_c_1 = [cl_1(ri_1_c1, :); cl_1(ri_1_c2, :); cl_1(ri_1_c3, :)];
% images_c_1 = [cd_1(ri_1_c1, :); cd_1(ri_1_c2, :); cd_1(ri_1_c3, :)];
% plotting_3_images(images_c_1, labels_c_1, col_size, row_size, 'Three randomly selected correctly classified digits for the 1NN-classifier using unclustered templates');
% % Picking three random wrongly classified digits to plot
% [ri_1_w1, ri_1_w2, ri_1_w3] = deal(randi(length(wl_1), 1), randi(length(wl_1), 1), randi(length(wl_1), 1));
% labels_w_1 = [wl_1(ri_1_w1, :); wl_1(ri_1_w2, :); wl_1(ri_1_w3, :)];
% images_w_1 = [wd_1(ri_1_w1, :); wd_1(ri_1_w2, :); wd_1(ri_1_w3, :)];
% plotting_3_images(images_w_1, labels_w_1, col_size, row_size, 'Three randomly selected wrongly classified digits for the 1NN-classifier using unclustered templates');
% 
% % 7NN
% % Picking three random correctly classified digits to plot
% [ri_1_c1_7, ri_1_c2_7, ri_1_c3_7] = deal(randi(length(cl_1_7), 1), randi(length(cl_1_7), 1), randi(length(cl_1_7), 1));
% labels_c_1_7 = [cl_1_7(ri_1_c1_7, :); cl_1_7(ri_1_c2_7, :); cl_1_7(ri_1_c3_7, :)];
% images_c_1_7 = [cd_1_7(ri_1_c1_7, :); cd_1_7(ri_1_c2_7, :); cd_1_7(ri_1_c3_7, :)];
% plotting_3_images(images_c_1_7, labels_c_1_7, col_size, row_size, 'Three randomly selected correctly classified digits for the 7NN-classifier using unclustered templates');
% Picking three random wrongly classified digits to plot
[ri_1_w1_7, ri_1_w2_7, ri_1_w3_7] = deal(randi(length(wl_1_7), 1), randi(length(wl_1_7), 1), randi(length(wl_1_7), 1));
labels_w_1_7 = [wl_1_7(ri_1_w1_7, :); wl_1_7(ri_1_w2_7, :); wl_1_7(ri_1_w3_7, :)];
images_w_1_7 = [wd_1_7(ri_1_w1_7, :); wd_1_7(ri_1_w2_7, :); wd_1_7(ri_1_w3_7, :)];
plotting_3_images(images_w_1_7, labels_w_1_7, col_size, row_size, 'Three randomly selected wrongly classified digits for the 7NN-classifier using unclustered templates');

disp("Ending task 1");
disp("-----------------------");

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
% % tp_2 : Vector that holds the true and predicted labels, used for plotting
% % The label matrices contain [True label, Predicted label]
% % Using deal to get all intialization on one line
% [w_2, c_2, cm_2, wd_2, wl_2, cd_2, cl_2, tp_lab_2] = deal(0, 0, zeros(10, 10), zeros(1, vec_size), zeros(1, 2), zeros(1, vec_size), zeros(1, 2), zeros(0, 2));
% 
% % 7-NN classifier:
% [w_2_7, c_2_7, cm_2_7, wd_2_7, wl_2_7, cd_2_7, cl_2_7, tp_lab_2_7] = deal(0, 0, zeros(10, 10), zeros(1, vec_size), zeros(1, 2), zeros(1, vec_size), zeros(1, 2), zeros(0, 2));
% 
% disp("Calculating distances and classifying");
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
%     distances_clustering = calculate_distance(testing_data, new_training_set);
%     
%     % 1-NN classifier
%     [cm_2, wd_2, wl_2, w_2, cd_2, cl_2, c_2, tp_lab_2] = classify_kNN(distances_clustering, number_of_tests, testing_data, testing_labels, training_labels, cm_2, wd_2, wl_2, w_2, cd_2, cl_2, c_2, 1, tp_lab_2);
%     % 7-NN classifier
%     [cm_2_7, wd_2_7, wl_2_7, w_2_7, cd_2_7, cl_2_7, c_2_7, tp_lab_2_7] = classify_kNN(distances_clustering, number_of_tests, testing_data, testing_labels, training_labels, cm_2_7, wd_2_7, wl_2_7, w_2_7, cd_2_7, cl_2_7, c_2_7, 7, tp_lab_2_7);
% end
% 
% error_rate_2 = w_2/num_test;
% disp("Error-rate for the 1NN-classifier for the clustered data: " + error_rate_2);
% error_rate_2_7 = w_2_7/num_test;
% disp("Error-rate for the 7NN-classifier for the clustered data: " + error_rate_2_7);

% % Plotting the confusion matrices
% plot_confusion_matrix(tp_lab_2, "Confusion matrix for the clustered data using the 1NN-classifier");
% pause(5);
% plot_confusion_matrix(tp_lab_2_7, "Confusion matrix for the clustered data using the 7NN-classifier");
% pause(5);
% 
% % 1NN
% % Picking three random correctly classified digits to plot
% [ri_2_c1, ri_2_c2, ri_2_c3] = deal(randi(length(cl_2), 1), randi(length(cl_2), 1), randi(length(cl_2), 1));
% labels_c_2 = [cl_2(ri_2_c1, :); cl_2(ri_2_c2, :); cl_2(ri_2_c3, :)];
% images_c_2 = [cd_2(ri_2_c1, :); cd_2(ri_2_c2, :); cd_2(ri_2_c3, :)];
% plotting_3_images(images_c_2, labels_c_2, col_size, row_size, 'Three randomly selected correctly classified digits for the 1NN-classifier using clustered templates');
% % Picking three random wrongly classified digits to plot
% [ri_2_w1, ri_2_w2, ri_2_w3] = deal(randi(length(wl_2), 1), randi(length(wl_2), 1), randi(length(wl_2), 1));
% labels_w_2 = [wl_2(ri_2_w1, :); wl_2(ri_2_w2, :); wl_2(ri_2_w3, :)];
% images_w_2 = [wd_2(ri_2_w1, :); wd_2(ri_2_w2, :); wd_2(ri_2_w3, :)];
% plotting_3_images(images_w_2, labels_w_2, col_size, row_size, 'Three randomly selected wrongly classified digits for the 1NN-classifier using clustered templates');
% 
% % 7NN
% % Picking three random correctly classified digits to plot
% [ri_2_c1_7, ri_2_c2_7, ri_2_c3_7] = deal(randi(length(cl_2_7), 1), randi(length(cl_2_7), 1), randi(length(cl_2_7), 1));
% labels_c_2_7 = [cl_2_7(ri_2_c1_7, :); cl_2_7(ri_2_c2_7, :); cl_2_7(ri_2_c3_7, :)];
% images_c_2_7 = [cd_2_7(ri_2_c1_7, :); cd_2_7(ri_2_c2_7, :); cd_2_7(ri_2_c3_7, :)];
% plotting_3_images(images_c_2_7, labels_c_2_7, col_size, row_size, 'Three randomly selected correctly classified digits for the 7NN-classifier using clustered templates');
% % Picking three random wrongly classified digits to plot
% [ri_2_w1_7, ri_2_w2_7, ri_2_w3_7] = deal(randi(length(wl_2_7), 1), randi(length(wl_2_7), 1), randi(length(wl_2_7), 1));
% labels_w_2_7 = [wl_2_7(ri_2_w1_7, :); wl_2_7(ri_2_w2_7, :); wl_2_7(ri_2_w3_7, :)];
% images_w_2_7 = [wd_2_7(ri_2_w1_7, :); wd_2_7(ri_2_w2_7, :); wd_2_7(ri_2_w3_7, :)];
% plotting_3_images(images_w_2_7, labels_w_2_7, col_size, row_size, 'Three randomly selected wrongly classified digits for the 7NN-classifier using clustered templates');
% 
% disp("Ending task 2");
% disp("-----------------------");

%%%------------------------------------
%%%------------------------------------
%%%         END OF TASK 2
%%%------------------------------------
%%%------------------------------------

% Classifying the test-vectors given the distance matrix using 1NN.
% Returns confusion-matrix, data, labels and amount for wrong and correct classification. 
function [cm, wd, wl, w, cd, cl, c, true_pred_lab] = classify_kNN(distances_set, num_test, test_data, test_labels, training_labels, cm, wd, wl, w, cd, cl, c, k, true_pred_lab)
    for l = 1:num_test
        [~, indices] = sort(distances_set(:, l));
        
        min_labels = zeros(k, 1);
        for y = 1:k
            min_labels(y) = training_labels(indices(y));
        end

        pl = mode(min_labels); % Finding the predicted label (Most frequent of the 7 labels)
        tl = test_labels(l); % Finding the true label
        
        true_pred_lab(end+1, :) = [tl, pl];
        
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
function [cm, wd, wl, w, cd, cl, c, true_pred_lab] = classify_1NN(distances_set, num_test, test_data, test_labels, training_labels, cm, wd, wl, w, cd, cl, c, true_pred_lab)
    for i = 1:num_test
        [~, index] = min(distances_set(:, i)); % Finds index in training set with min. dist.
        pl = training_labels(index); % Predicted/Classified label
        tl = test_labels(i); % True label
        true_pred_lab(end+1, :) = [tl, pl]; % Adding true and predicted labels to array

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
function plotting_images(image_data, labels, col_size, row_size)
    image(transpose(reshape(image_data, col_size, row_size)));
    title("True label - " + labels(1, 1) + " / Predicted label - " + labels(1, 2));
end

function plotting_3_images(image_datas, labels, col_size, row_size, title_overall)
    figure('Position', [100 100 800 300]);
    subplot('Position', [0.05 0.1 0.25 0.8]);
    imshow(transpose(reshape(image_datas(1, :), col_size, row_size)), []);
    title("True label - " + labels(1, 1) + " / Predicted label - " + labels(1, 2));
    subplot('Position', [0.35 0.1 0.25 0.8]);
    imshow(transpose(reshape(image_datas(2, :), col_size, row_size)), []);
    title("True label - " + labels(2, 1) + " / Predicted label - " + labels(2, 2));
    subplot('Position', [0.65 0.1 0.25 0.8]);
    imshow(transpose(reshape(image_datas(3, :), col_size, row_size)), []);
    title("True label - " + labels(3, 1) + " / Predicted label - " + labels(3, 2));

    % Overall title
    sgtitle(title_overall);
end

% Calculates the distances given test-vector and templates
function distances_return = calculate_distance(test_set, templates)
    distances_return = dist(templates, transpose(test_set));
end

function s = sorting(training_data, training_label, num_train, vec_size)
    s = {zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size), zeros(0, vec_size)};
    for j = 1:num_train
        s{training_label(j)+1}(end+1, :) = training_data(j, :);
    end
end

function plot_confusion_matrix(true_pred_labels, title_cm)
    confusionchart(true_pred_labels(:, 1), true_pred_labels(:, 2));
    title(title_cm);
end
