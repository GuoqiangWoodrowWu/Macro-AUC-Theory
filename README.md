# Towards Understanding Generalization of Macro-AUC in Multi-label Learning
This repository is the official implementation of "Guoqiang Wu, Chongxuan Li and Yilong Yin. Towards Understanding Generalization of Macro-AUC in Multi-label Learning" accepted in ICML 2023.
## Programming Language
The source code is written by Matlab
## File description
- ./Datasets -- the benchmarks datasets downloaded from the websites http://mulan.sourceforge.net/datasets-mlc.html and http://palm.seu.edu.cn/zhangml/
- ./measures -- the measures for multi-label learning on Maro-AUC
- ./Results -- store the experimental results
- ./CrossValidation.m -- used to create cross-validation data
- ./train_logistic_label_wise_pairwise_SVRG_BB.m -- utilize SVRG-BB to train the model with surrogate pairwise loss (i.e. A^{pa}) where the base loss is logistic loss
- ./train_logistic_cost_sensitive_SVRG_BB.m -- utilize SVRG-BB to train the model with different surrogate univariate losses (including A^{u_k}, k = 1,2.) where the base loss is logistic loss
- ./calculate_cost_matrix.m -- calculate the cost matrix for corresponding univarite loss (including L_{u_k}, k = 1,2.)
- ./Predict_score.m -- predict the score function
- ./Evaluation_Metrics.m -- evaluate the model on Macro-AUC measure
- run_linear_pa.m -- run the code to evaluate A^{pa}
- run_linear_u1.m -- run the code to evaluate A^{u_1}
- run_linear_u2.m -- run the code to evaluate A^{u_2}
- plot_label_wise_imbalance.m -- plot label-wise class imbalance  
## Run
Run the run_linear_pa.m, run_linear_u1.m, run_linear_u2.m, run_linear_u3.m and run_linear_u4.m in MATLAB, and it will run as its default parameters on sample datasets.
