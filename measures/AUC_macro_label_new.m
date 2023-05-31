function [ AUCMacro_label ] = AUC_macro_label_new( outputs, test_target )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [num_instance, num_class] = size(outputs);
    test_target(test_target < 1) = -1;
    
    AUCMacro_label = 0.0;
    for j = 1: num_class
        p_list = find(test_target(:, j) > 0);
        q_list = find(test_target(:, j) < 0);
        pos_num = length(p_list);
        neg_num = length(q_list);
        
        if pos_num == 0 || neg_num == 0
            continue;
        end
        
        correct_num = 0;
        for p = 1: pos_num
            for q = 1: neg_num
                if outputs(p_list(p), j) >= outputs(q_list(q), j)
                    correct_num = correct_num + 1;
                end
            end
        end
        auc_one_class = correct_num / (pos_num * neg_num);
        AUCMacro_label = AUCMacro_label + auc_one_class;
    end
    AUCMacro_label = AUCMacro_label / num_class;
end

