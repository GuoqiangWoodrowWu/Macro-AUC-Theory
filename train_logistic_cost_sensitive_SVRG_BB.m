function [W, obj] = train_logistic_cost_sensitive_SVRG_BB( X, Y, lambda, alpha, option )
%   Optimize macro-auc with its surrogate loss, base loss function: logistic loss
%   alpha: learning_rate
%   lambda_1 for l2 norm
    
    [num_instance, num_feature] = size(X);
    num_label = size(Y, 2);
    W = zeros(num_feature, num_label);
    
    C = calculate_cost_matrix(Y, option);
    
    % Do serveral SGD steps first
    for i = 1: 10
        index = randi(num_instance);
        GD_one = calculate_one_gradient(X(index,:), Y(index,:), C(index,:), W, lambda);
        W = W - alpha * GD_one;
    end
    
%     num_s = 10;
    num_s = 20;
    m = 2 * num_instance;
    epsilon = 0;
    
    for i = 1: num_s
        W1 = W;
        fG1 = calculate_all_gradient(X, Y, C, W1, lambda);
        if i > 1
            if i > 2 && abs(obj(i-1, 1) - obj(i-2, 1)) / obj(i-2, 1) <= epsilon
                break;
            end
            alpha = norm(W1-W0, 'fro')^2 / trace((W1-W0)'*(fG1-fG0)) / m;
            fprintf('alpha: %.5f\n', alpha);
            % If alpha is too big, this may cause numerical problem (e.g.,
            % NAN). Thus, control to make it not to big.
%             if alpha > 0.05
%                 alpha = 0.01;
%             end
        end
        fG0 = fG1;
        W0 = W1;
        for j = 1: m
            index = randi(num_instance);
            GD_one = calculate_one_gradient(X(index,:), Y(index,:), C(index,:), W, lambda);
            GD_ = calculate_one_gradient(X(index,:), Y(index,:), C(index,:), W1, lambda);
            W = W - alpha * (GD_one - GD_ + fG1);
            if isnan(W)
               disp('here0');
               return;
            end
        end
        obj(i,1) = calculate_objective_function(X, Y, C, W, lambda);
        fprintf('Step %d: the objective function value is %.5f\n', i, obj(i,1));
    end
end

function [f_value] = calculate_objective_function(X, Y, C, W, lambda)
    f_value = 0.5 * lambda * norm(W, 'fro')^2;
    [num_instance, num_class] = size(Y);
    
    I = ones(num_instance, num_class);
    %Z = zeros(num_instance, num_class);
    temp1 = log(I + exp(-Y .* (X * W))) .* C;
    if isnan(W)
        fprintf('here1');
    end
    if isnan(temp1)
        fprintf('here2');
    end
    f_value_point = sum(sum(temp1, 2));
    %f_value_point = f_value_point / num_class;
    
    f_value = f_value + 1 / num_instance * f_value_point;
end


function [grad] = calculate_all_gradient(X, Y, C, W, lambda)
    [num_instance, num_class] = size(Y);
    num_feature = size(X, 2);

    grad = lambda * W;
    Z_m = zeros(num_feature, num_class);
    
    grad_point = Z_m;
    I = ones(num_instance, num_class);
    %Z = zeros(num_instance, num_class);
    %grad_point = X' * (-Y .* C .* sign(max(Z, I - Y .* (X * W))));
    tmp = exp(-Y .* (X * W));
    grad_point = X' * (-Y .* C .* tmp ./ (I + tmp));
    %grad_point = grad_point / num_class;
    
    if isnan(tmp)
        fprintf('here3');
    end
    if isnan(grad_point)
        fprintf('here4');
    end
    
    grad = grad + grad_point / num_instance;
end

function [grad_one] = calculate_one_gradient(x, y, c, W, lambda)
% input: size(x) = [1, num_feature], size(y) = [1, num_class]
% Calculate hinge loss gradient
    [num_feature, num_class] = size(W);
    Z_m = zeros(num_feature, num_class);
    grad_one = lambda * W;
    
    grad_point = Z_m;

    I = ones(1, num_class);
    Z = zeros(1, num_class);
    tmp = exp(- y .* (x * W));
    grad_point = x' * (-y .* c .* tmp ./ (I + tmp));
    % add 
    %grad_point = grad_point / num_class;
    
    if isnan(tmp)
        fprintf('here4');
    end
    if isnan(grad_point)
        fprintf('here5');
    end

    grad_one = grad_one + grad_point;

end