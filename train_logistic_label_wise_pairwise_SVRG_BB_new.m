function [ W ] = train_logistic_label_wise_pairwise_SVRG_BB_new( X, Y, lambda, alpha )
%UNTITLED2 Summary of this function goes here
%   paiwise surrogate loss with base loss is logistic
%   alpha: learning_rate
%   lambda_1 for l2 norm
    [num_instance, num_feature] = size(X);
    num_label = size(Y, 2);
    W = zeros(num_feature, num_label);
    
    
    for j = 1: num_label
        fprintf('Training the label %d:\n', j);
        W(:,j) = train_model_for_one_label(X, Y(:,j), lambda, alpha);
    end
   
end

function [w] = train_model_for_one_label(X, y, lambda, alpha)
    [num_instance, num_feature] = size(X);
    w = zeros(num_feature, 1);
    
    p_list = find(y > 0);
    q_list = find(y < 0);
    num_pos = length(p_list);
    num_neg = length(q_list);
    
    if num_pos == 0 || num_neg == 0
        return;
    end
    
    % Do serveral SGD steps first
    for i = 1: 10
        pos_index_tmp = randi(num_pos);
        pos_index = p_list(pos_index_tmp);
        neg_index_tmp = randi(num_neg);
        neg_index = q_list(neg_index_tmp);
        
        GD_one = calculate_one_gradient(X, pos_index, neg_index, w, lambda);
        w = w - alpha * GD_one;
%         size(w)
    end
    
    num_s = 30;
    %num_s = 0;
    obj = zeros(num_s, 1);
    m = 2 * (num_pos * num_neg);
    epsilon = 10^-6;
    for i = 1: num_s
        w1 = w;
        fG1 = calculate_all_gradient(X, p_list, q_list, w1, lambda);
        if i > 1
            if i > 2 && abs(obj(i-1, 1) - obj(i-2, 1)) / obj(i-2, 1) <= epsilon
                break;
            end
            alpha = norm(w1-w0, 'fro')^2 / trace((w1-w0)'*(fG1-fG0)) / m;
        end
        fG0 = fG1;
        w0 = w1;
        for j = 1: m
            pos_index_tmp = randi(num_pos);
            pos_index = p_list(pos_index_tmp);
            neg_index_tmp = randi(num_neg);
            neg_index = q_list(neg_index_tmp);
            
            GD_one = calculate_one_gradient(X, pos_index, neg_index, w, lambda);
            GD_ = calculate_one_gradient(X, pos_index, neg_index, w1, lambda);
            w = w - alpha * (GD_one - GD_ + fG1);
            %if isnan(W)
            %    return;
            %end
        end
        obj(i,1) = calculate_objective_function(X, p_list, q_list, w, lambda);
        fprintf('Step %d: the objective function value is %.5f\n', i, obj(i,1));
    end
end

function [f_value] = calculate_objective_function(X, p_list, q_list, w, lambda)
    f_value = 0.5 * lambda * norm(w, 'fro')^2;

    f_value_loss = 0;
    pos_num = length(p_list);
    neg_num = length(q_list);
    for p = pos_num
        for q = neg_num
            % hinge loss
%             f_value_loss = f_value_loss + max(0, 1 - dot(w, X(p_list(p),:) - X(q_list(q),:)));
            % logistic loss
            f_value_loss = f_value_loss + log(1 + exp(- dot(w, X(p_list(p),:) - X(q_list(q),:))));
        end
    end

    f_value = f_value + f_value_loss / (pos_num * neg_num);
end


function [grad] = calculate_all_gradient(X, p_list, q_list, w, lambda)
    num_feature = size(X, 2);

    grad = lambda * w;
    Z_m = zeros(num_feature, 1);
    
    grad_loss = Z_m;
    pos_num = length(p_list);
    neg_num = length(q_list);
    
    for p = 1: pos_num
        for q = 1: neg_num
            % logistic loss
            grad_loss = grad_loss + (X(q_list(q),:) - X(p_list(p),:))' / (1 + exp(dot(w, X(p_list(p),:) - X(q_list(q),:))));
            % hinge loss
            tmp_grad = Z_m;
            if dot(w, X(p_list(p),:) - X(q_list(q),:)) <= 1
                tmp_grad = - (X(p_list(p),:) - X(q_list(q),:))';
            end
            grad_loss = grad_loss + tmp_grad;
        end
    end
    
    grad = grad + grad_loss / (pos_num * neg_num);
end

function [grad_one] = calculate_one_gradient(X, pos_index, neg_index, w, lambda)
% input: size(x) = [1, num_feature], size(y) = [1, num_class]
% Calculate logistic loss gradient
%     [num_feature, num_class] = size(W);
%     Z_m = zeros(num_feature, num_class);
    grad_one = lambda * w;
    % logistic loss
    grad_rank = (X(neg_index,:) - X(pos_index,:))' ./ (1 + exp(dot(w, X(pos_index,:) - X(neg_index,:))));

    % hinge loss
%     num_feature = size(X, 2);
%     grad_rank = zeros(num_feature, 1);
%     if dot(w, X(pos_index,:) - X(neg_index,:)) <= 1
%         grad_rank = -(X(pos_index,:) - X(neg_index,:))';
%     end
    grad_one = grad_one + grad_rank;
end
