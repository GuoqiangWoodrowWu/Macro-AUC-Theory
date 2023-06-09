function [ AUCMacro_label_detail ] = AUC_macro_label_details( outputs, test_target )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [num_instance, num_class] = size(outputs);
    test_target(test_target < 1) = -1;
    
    AUCMacro_label_detail = zeros(1, num_class);
    for i = 1: num_class
        pos_num = sum(test_target(:, i) == 1);
        neg_num = num_instance - pos_num;
        
        if pos_num == 0 || neg_num == 0
            continue;
        end
        
        correct_num = 0;
        for p = 1: num_instance
            for q = (p + 1): num_instance
                if test_target(p, i) == 1 && test_target(q, i) == -1
%                     if outputs(p, i) >= outputs(q, i)
                    if outputs(p, i) > outputs(q, i)
                        correct_num = correct_num + 1;
                    end
                elseif test_target(p, i) == -1 && test_target(q, i) == 1
%                     if outputs(q, i) >= outputs(p, i)
                    if outputs(q, i) > outputs(p, i)
                        correct_num = correct_num + 1;
                    end
                end
            end
        end
        auc_one_class = correct_num / (pos_num * neg_num);
        AUCMacro_label_detail(i) = auc_one_class;
    end
end

