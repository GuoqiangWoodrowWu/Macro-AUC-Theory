
% Y = S.target;
Y = target;
% Y = Y_train; 
Y(Y < 1) = 0;
[num_instance, num_label] = size(Y);

imbalance_level = size(1, num_label);

for j = 1: num_label
    positive_num = sum(Y(:, j));
%     imbalance_level(1, j) = positive_num; 
    imbalance_level(1, j) = min(positive_num, num_instance - positive_num) / num_instance; 
end
% hist(imbalance_level);
plot(imbalance_level, 'b-o');
set(gca, 'FontSize', 18);
axis([-inf inf 0 0.5]);
xlabel('Label Index (k)', 'fontsize', 18);
ylabel('Value (\tau_k)', 'fontsize', 18);

sum_value1 = 0;
sum_value2 = 0;
for j = 1: num_label
    sum_value1 = sum_value1 + 1 / sqrt(imbalance_level(1, j));
    sum_value2 = sum_value2 + 1 / imbalance_level(1, j);
end
mean_value1 = sum_value1 / num_label;
mean_value2 = sqrt(sum_value2 / num_label);
min_value = 1 / min(imbalance_level);
mean_times_min = mean_value1 * min_value;