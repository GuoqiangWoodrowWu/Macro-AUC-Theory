clear;
close all;
clc;

% dataset name
dataset_name = 'CAL500';
path_main = pwd;
path_save = strcat(path_main, filesep, 'Results');
path_data = strcat(path_main, filesep, 'Datasets');
file_save = strcat(path_save, filesep, 'Results_linear_u1_CAL500.csv');

% setting
rand('seed', 2^40);
result = [];
K_fold = 3; % cross-validation: 3-fold
lambda = 0.001;

surrogate_loss_option = 'surrogate_loss_u1';

% Loading the dataset
file_name = strcat(path_data, filesep, dataset_name, '.mat');
S = load(file_name);

X_all = S.data;
Y_all = S.target;
Y_all(Y_all < 1) = -1;
% append one feature with all equal to 1 to correspond to the bias
num_feature_origin = size(X_all, 2);
X_all(:, num_feature_origin + 1) = 1;

%normalization
[X_all, PS] = mapstd(X_all', 0, 1);
X_all = X_all';

% Shuffle the dataset
[num_samples, num_feature] = size(X_all);
shuffle_index = randperm(num_samples);
X_all = X_all(shuffle_index, :);
Y_all = Y_all(shuffle_index, :);

% Do cross-validation
hl = zeros(1, K_fold);
sa = zeros(1, K_fold);
auc_macro = zeros(1, K_fold);

tic;
for index_cv = 1: K_fold
    [X_train, Y_train, X_vali, Y_vali] = CrossValidation(X_all, Y_all, K_fold, index_cv);      
    % train the train dataset and predict the test dataset 
    alpha = 0.01;
    [ W, obj ] = train_logistic_cost_sensitive_SVRG_BB( X_train, Y_train, lambda, alpha, surrogate_loss_option );
    [ pre_F_vali ] = Predict_score( X_vali, W );

    [ AUCMacro_label ] = Evaluation_Metrics( pre_F_vali, Y_vali );
    
    [ pre_F_train ] = Predict_score( X_train, W );
    [ AUCMacro_label_train ] = Evaluation_Metrics( pre_F_train, Y_train );
    
    [ AUCMacro_label_detail ] = AUC_macro_label_details( pre_F_vali, Y_vali );

    auc_macro(index_cv) = AUCMacro_label;

end
toc;
time = double(toc);

AUCMacro_LABEL_cv_mean = mean(auc_macro);
AUCMacro_LABEL_cv_std = std(auc_macro);
result = [result; AUCMacro_LABEL_cv_mean AUCMacro_LABEL_cv_std lambda time];
csvwrite(file_save, result);
