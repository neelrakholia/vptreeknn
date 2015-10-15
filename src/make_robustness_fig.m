


close all;

data = load('mnist_hyptan_robust.txt');

powers = 0:12;

plot(powers, data(:,1), 'r-', 'LineWidth', 6);
hold on;
plot(powers, data(:,2), 'k-.', 'LineWidth', 6);
plot(powers, data(:,3), 'b--', 'LineWidth', 6);

xlabel('h_H (in powers of 10)', 'FontSize', 22);
ylabel('n_s', 'FontSize', 22);

hleg = legend('VP Tree', 'KLSH', 'MKS', 'Location', 'NorthEast');
set(hleg, 'FontSize', 22);
set(gca, 'FontSize', 22);
axis([0,12,-inf,inf])

saveas(gcf, 'mnist_hyptan_robust.pdf');

figure();

data = load('uniform_hyptan_robust.txt');

powers = -2:5;

plot(powers, data(:,1), 'r-', 'LineWidth', 6);
hold on;
plot(powers, data(:,2), 'k-.', 'LineWidth', 6);
plot(powers, data(:,3), 'b--', 'LineWidth', 6);

xlabel('h_H (in powers of 10)', 'FontSize', 22);
ylabel('n_s', 'FontSize', 22);

hleg = legend('VP Tree', 'KLSH', 'MKS', 'Location', 'NorthEast');
set(hleg, 'FontSize', 22);
set(gca, 'FontSize', 22);
axis([-inf inf 0, 1.01])

saveas(gcf, 'uniform_hyptan_robust.pdf');

