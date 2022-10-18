%% READ DATA FROM CSV FILE
data_idx = 3;
data = csvread('../results/data_'+string(data_idx)+'/data_'+string(data_idx)+'.csv');

n = 37;
m = 13;
N = (length(data) - n) / (n + m) + 1;

pb = zeros(3, N);
Theta = zeros(3, N);
p1 = zeros(3, N);
p2 = zeros(3, N);
p3 = zeros(3, N);
p4 = zeros(3, N);
t = zeros(1, N);

F1 = zeros(3, N-1);
F2 = zeros(3, N-1);
F3 = zeros(3, N-1);
F4 = zeros(3, N-1);

for k = 1:N
    pb(:, k) = data(1+(n + m)*(k - 1):3+(n + m)*(k - 1));
    Theta(:, k) = data(4+(n + m)*(k - 1):6+(n + m)*(k - 1));
    p1(:, k) = data(7+(n + m)*(k - 1):9+(n + m)*(k - 1));
    p2(:, k) = data(10+(n + m)*(k - 1):12+(n + m)*(k - 1));
    p3(:, k) = data(13+(n + m)*(k - 1):15+(n + m)*(k - 1));
    p4(:, k) = data(16+(n + m)*(k - 1):18+(n + m)*(k - 1));
    t(k) = data(n+(n + m)*(k - 1));
    if k < N
        F1(:, k) = data(n+1+(n + m)*(k - 1):n+3+(n + m)*(k - 1));
        F2(:, k) = data(n+4+(n + m)*(k - 1):n+6+(n + m)*(k - 1));
        F3(:, k) = data(n+7+(n + m)*(k - 1):n+9+(n + m)*(k - 1));
        F4(:, k) = data(n+10+(n + m)*(k - 1):n+12+(n + m)*(k - 1));
    end
end

%% DATA INTERPOLATION
new_t = 0:0.005:t(N);

new_xb = interp1(t, pb(1, :), new_t);
new_yb = interp1(t, pb(2, :), new_t);
new_zb = interp1(t, pb(3, :), new_t);

new_roll = interp1(t, Theta(1, :), new_t);
new_pitch = interp1(t, Theta(2, :), new_t);
new_yaw = interp1(t, Theta(3, :), new_t);

new_x1 = interp1(t, p1(1, :), new_t);
new_y1 = interp1(t, p1(2, :), new_t);
new_z1 = interp1(t, p1(3, :), new_t);

new_x2 = interp1(t, p2(1, :), new_t);
new_y2 = interp1(t, p2(2, :), new_t);
new_z2 = interp1(t, p2(3, :), new_t);

new_x3 = interp1(t, p3(1, :), new_t);
new_y3 = interp1(t, p3(2, :), new_t);
new_z3 = interp1(t, p3(3, :), new_t);

new_x4 = interp1(t, p4(1, :), new_t);
new_y4 = interp1(t, p4(2, :), new_t);
new_z4 = interp1(t, p4(3, :), new_t);

new_F1x = interp1(t(1:N-1), F1(1, :), new_t);
new_F1y = interp1(t(1:N-1), F1(2, :), new_t);
new_F1z = interp1(t(1:N-1), F1(3, :), new_t);

new_F2x = interp1(t(1:N-1), F2(1, :), new_t);
new_F2y = interp1(t(1:N-1), F2(2, :), new_t);
new_F2z = interp1(t(1:N-1), F2(3, :), new_t);

new_F3x = interp1(t(1:N-1), F3(1, :), new_t);
new_F3y = interp1(t(1:N-1), F3(2, :), new_t);
new_F3z = interp1(t(1:N-1), F3(3, :), new_t);

new_F4x = interp1(t(1:N-1), F4(1, :), new_t);
new_F4y = interp1(t(1:N-1), F4(2, :), new_t);
new_F4z = interp1(t(1:N-1), F4(3, :), new_t);

%% PLOT
fig = figure(1);
v = VideoWriter('../results/data_'+string(data_idx)+'/landing');
open(v);
plot_force = true;
show_body_pos = true;
alpha = 0.003;

for k = 2:length(new_t) - 4
    clf;
    set(fig, 'position', [400, 400, 1200, 900])

    % plot axes of the world frame (rgb)
    plot3([0, 0.1], [0, 0], [0, 0], 'red', 'LineWidth', 2), hold on;
    plot3([0, 0], [0, 0.1], [0, 0], 'green', 'LineWidth', 2), hold on;
    plot3([0, 0], [0, 0], [0, 0.1], 'blue', 'LineWidth', 2), hold on;

    % plot axes of the body frame (rgb)
    plotFrame([new_xb(k); new_yb(k); new_zb(k)], ...
        new_roll(k), new_pitch(k), new_yaw(k));

    % plot robot
    plot3([new_x1(k), new_xb(k)], ...
        [new_y1(k), new_yb(k)], ...
        [new_z1(k), new_zb(k)]), hold on;
    plot3([new_x2(k), new_xb(k)], ...
        [new_y2(k), new_yb(k)], ...
        [new_z2(k), new_zb(k)]), hold on;
    plot3([new_x3(k), new_xb(k)], ...
        [new_y3(k), new_yb(k)], ...
        [new_z3(k), new_zb(k)]), hold on;
    plot3([new_x4(k), new_xb(k)], ...
        [new_y4(k), new_yb(k)], ...
        [new_z4(k), new_zb(k)]), hold on;

    % plot feet
    plot3([new_x1(k)], [new_y1(k)], [new_z1(k)], 'o'), hold on;
    plot3([new_x2(k)], [new_y2(k)], [new_z2(k)], 'o'), hold on;
    plot3([new_x3(k)], [new_y3(k)], [new_z3(k)], 'o'), hold on;
    plot3([new_x4(k)], [new_y4(k)], [new_z4(k)], 'o'), hold on;
    
    % plot forces
    if plot_force
        plot3([new_x1(k), new_x1(k) + alpha * new_F1x(k)], ...
            [new_y1(k), new_y1(k) + alpha * new_F1y(k)], ...
            [new_z1(k), new_z1(k) + alpha * new_F1z(k)]), hold on;
        plot3([new_x2(k), new_x2(k) + alpha * new_F2x(k)], ...
            [new_y2(k), new_y2(k) + alpha * new_F2y(k)], ...
            [new_z2(k), new_z2(k) + alpha * new_F2z(k)]), hold on;
        plot3([new_x3(k), new_x3(k) + alpha * new_F3x(k)], ...
            [new_y3(k), new_y3(k) + alpha * new_F3y(k)], ...
            [new_z3(k), new_z3(k) + alpha * new_F3z(k)]), hold on;
        plot3([new_x4(k), new_x4(k) + alpha * new_F4x(k)], ...
            [new_y4(k), new_y4(k) + alpha * new_F4y(k)], ...
            [new_z4(k), new_z4(k) + alpha * new_F4z(k)]), hold on;
    end

    if show_body_pos
        body_coordinate = 'Body COM Coordinate (m): (' ...
            +string(round(new_xb(k), 3)) + ',' ...
            +string(round(new_yb(k), 3)) + ',' ...
            +string(round(new_zb(k), 3)) + ')';
        text(new_xb(k)+0.01, new_yb(k)-0.01, new_zb(k), body_coordinate);
        
        body_orientation = 'Body RPY (deg): (' ...
            +string(round(rad2deg(new_roll(k)), 3)) + ',' ...
            +string(round(rad2deg(new_pitch(k)), 3)) + ',' ...
            +string(round(rad2deg(new_yaw(k)), 3)) + ')';
        text(new_xb(k)+0.01, new_yb(k)-0.01, new_zb(k) - 0.05, body_orientation);
    end

    axis equal, grid on;

    xlim([-0.5, 0.5]);
    ylim([-0.5, 0.5]);
    zlim([-0.1, 1]);

    frame = getframe(fig);
    writeVideo(v, frame);
end

close(v);