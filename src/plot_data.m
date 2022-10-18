data = csvread('../results/data_3/data_3.csv');
N = 41;

% plot forces on the feet
F1 = zeros(N-1, 3);
F2 = zeros(N-1, 3);
F3 = zeros(N-1, 3);
F4 = zeros(N-1, 3);

F1_mag = zeros(N-1, 1);
F2_mag = zeros(N-1, 1);
F3_mag = zeros(N-1, 1);
F4_mag = zeros(N-1, 1);

t = zeros(N-1, 1);

for k = 1:N - 1
    t(k) = data(37+50*(k - 1));
    F1(k, :) = data(38+50*(k - 1):40+50*(k - 1));
    F2(k, :) = data(41+50*(k - 1):43+50*(k - 1));
    F3(k, :) = data(44+50*(k - 1):46+50*(k - 1));
    F4(k, :) = data(47+50*(k - 1):49+50*(k - 1));
    
    F1_mag(k) = norm(F1(k, :));
    F2_mag(k) = norm(F2(k, :));
    F3_mag(k) = norm(F3(k, :));
    F4_mag(k) = norm(F4(k, :));
end

new_t = linspace(0, t(N-1));
new_F1_mag = interp1(t, F1_mag, new_t);

figure(1);
plot(t, F1_mag), hold on;
plot(t, F2_mag), hold on;
plot(t, F3_mag), hold on;
plot(t, F4_mag), hold on;

xlim([0, t(N-1)]);

grid on;