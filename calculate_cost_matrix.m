function [C] = calculate_cost_matrix(Y, option)
% Calculate the cost matrix for different measures.
%   Detailed explanation goes here
[num_instance, num_label] = size(Y);
C = ones(num_instance, num_label);
Y(Y < 1) = 0;
if strcmp(option, 'surrogate_loss_u1')
    %C = ones(num_instance, num_label);
    for i = 1: num_instance
        C(i, :) = C(i, :) ./ num_label;
    end
elseif strcmp(option, 'surrogate_loss_u2')
    %C = ones(num_instance, num_label);
    for j = 1: num_label
        tmp_positive = sum(Y(:,j));
        tmp_negative = num_instance - tmp_positive;
        if tmp_positive == 0 || tmp_negative == 0
            C(:,j) = 1;
            continue;
        end
        for i = 1: num_instance
            if Y(i, j) == 1
                C(i, j) = 1 / tmp_positive;
            else
                C(i, j) = 1 / tmp_negative;
            end
        end
    end
    C = C .* (num_instance / num_label);
end
    
end

