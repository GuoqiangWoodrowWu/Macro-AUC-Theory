function [ AUCmacro_label ] = Evaluation_Metrics( pre_F, Y )
%UNTITLED4 Evaluate the model for many metrics
%   Detailed explanation goes here

    cd('./measures');
%     Ranking_Loss = Ranking_loss(pre_F, Y);
    AUCmacro_label = AUC_macro_label_new(pre_F, Y);
    cd('../');
end
