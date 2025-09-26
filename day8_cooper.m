function day8_cooper()
% Solve the Jansen-leg linkage for a given crank angle and plot the result
% using link segments from the link-to-vertex list (matches handout)

    %parameters (from handout) 
    leg_params = build_leg_params();

    % choose a crank angle (radians)
    theta = 0.0;

    % initial guess for [x1;y1; x2;y2; ... ; x7;y7] 
    vertex_guess_coords = [ ...
          0;   50;   % v1
        -50;    0;   % v2
        -50;   50;   % v3
        -100;   0;   % v4
        -100; -50;   % v5
         -50; -50;   % v6
         -50;-100];  % v7

    %compute a root of all constraints at this theta
    V_root = compute_coords(vertex_guess_coords, leg_params, theta);

    % sanity check – all constraints should be ~0
    res = linkage_error_func(V_root, leg_params, theta);
    fprintf('||constraint residual|| = %.3e\n', norm(res));

    % plot using link list 
    V = column_to_matrix(V_root);         % N x 2
    L = leg_params.link_to_vertex_list;   % m x 2 (link i connects vertices L(i,1) and L(i,2))

    figure; clf; hold on; axis equal;
    % draw links
    for i = 1:size(L,1)
        a = L(i,1); b = L(i,2);
        plot([V(a,1), V(b,1)], [V(a,2), V(b,2)], 'k-', 'LineWidth', 2);
    end
    % draw vertices
    plot(V(:,1), V(:,2), 'ro', 'MarkerFaceColor','r', 'MarkerSize', 6);

    % draw crank circle & current crank radius (nice visual)
    t = linspace(0, 2*pi, 256);
    c = leg_params.vertex_pos0(:).';
    R = leg_params.crank_length;
    plot(c(1)+R*cos(t), c(2)+R*sin(t), 'k--', 'LineWidth', 1);
    plot([c(1), c(1)+R*cos(theta)], [c(2), c(2)+R*sin(theta)], 'k-', 'LineWidth', 1.5);

    grid on;
    title('Solved linkage (links follow link\_to\_vertex\_list)');
    xlabel('x'); ylabel('y');

    %velocity plots
    figure;
    thetas = 0:0.005:2*pi;
    n = length(thetas);
    V_rootold = V_root;
    linxs = zeros(n,1);
    linys = zeros(n,1);
    Vroots = zeros(n,14);
    for i = 1:n
        V_rootnew = compute_coords(V_rootold, leg_params, theta);
        Vroots(i, :) = V_rootnew';
        dVdtheta = compute_velocities(V_rootnew, leg_params, theta);
        linxs(i) = dVdtheta(13);
        linys(i) = dVdtheta(14);
    end
    compxs = zeros(n,1);
    compys = zeros(n,1);
    col13 = V_roots(:, 13);
    col14 = V_roots(:, 14);
    
    subplot(2,1,1)
    plot(thetas, linxs)

    subplot(2,1,2)
    plot(thetas, linys)
end

%  CORE SOLVER 

function [root, it, flag, glist] = multi_newton_solver(FUN_both, X0, Athresh, Bthresh, maxit, numdiff)
% General multidimensional Newton’s method.
% FUN_both(X): either returns F(X)            (numdiff==1)   OR
%                 returns [F(X), J(X)]          (numdiff==2)
% Early termination: ||ΔX|| < Athresh OR ||F|| < Bthresh
% Safeguard: stop if Jacobian is (near) singular
% Outputs:
%     root: solution vector
%     it  : iteration count
%     flag: 1 success, -2 bad/singular Jacobian
%     glist: columns are the iterate history

    glist = [];
    root  = X0;
    it    = 0;
    flag  = 0;

    glist(:,end+1) = root;

    % get residual & Jacobian
    if numdiff == 2
        try
            [FX, J] = FUN_both(root);
        catch
            FX = FUN_both(root);
            J  = approximate_jacobian(FUN_both, root);
        end
    else
        FX = FUN_both(root);
        J  = approximate_jacobian(FUN_both, root);
    end

    while norm(FX) > Bthresh && it < maxit
        % singularity guard (use J*J' to handle non-square cases robustly)
        if abs(det(J*J')) < eps
            flag = -2; return;
        end

        X_new = root - J\FX;
        glist(:,end+1) = X_new;

        if norm(X_new - root) < Athresh
            root = X_new; flag = 1; return
        end

        root = X_new;
        if numdiff == 2
            try
                [FX, J] = FUN_both(root);
            catch
                FX = FUN_both(root);
                J  = approximate_jacobian(FUN_both, root);
            end
        else
            FX = FUN_both(root);
            J  = approximate_jacobian(FUN_both, root);
        end
        it = it + 1;
    end

    flag = 1; % reached residual threshold or maxit
end

function J = approximate_jacobian(FUN, X)
% Centered finite-difference Jacobian: J(:,i) ≈ (F(x+h/2*ei)-F(x-h/2*ei))/h
    h = 1e-6;
    n = numel(X);
    F0 = FUN(X);
    m = numel(F0);
    J = zeros(m, n);
    for i = 1:n
        ei      = zeros(n,1); ei(i) = h/2;
        F_plus  = FUN(X + ei);
        F_minus = FUN(X - ei);
        J(:,i)  = (F_plus - F_minus) / h;
    end
end

% LINKAGE SETUP
function leg_params = build_leg_params()
% All constants exactly as described in the handout

    leg_params = struct();

    % counts
    leg_params.num_vertices = 7;
    leg_params.num_linkages = 10;

    % link-to-vertex list (each row = [vertex_a vertex_b]) 
    leg_params.link_to_vertex_list = [ ...
        1 3;   % link 1
        3 4;   % link 2
        2 3;   % link 3
        2 4;   % link 4
        4 5;   % link 5
        2 6;   % link 6
        1 6;   % link 7
        5 6;   % link 8
        5 7;   % link 9
        6 7];  % link 10

    %link lengths (same order as rows above) 
    leg_params.link_lengths = [ ...
         50.0;  % link 1
         55.8;  % link 2
         41.5;  % link 3
         40.1;  % link 4
         39.4;  % link 5
         39.3;  % link 6
         61.9;  % link 7
         36.7;  % link 8
         65.7;  % link 9
         49.0]; % link 10

    % additional constraints
    leg_params.crank_length = 15.0;            % radius of vertex 1 from crank center (vertex 0)
    leg_params.vertex_pos0  = [0; 0];          % fixed crank center
    leg_params.vertex_pos2  = [-38.0; -7.8];   % fixed vertex 2
end

% UTILITIES

function M = column_to_matrix(coords_col)
% [x1;y1;x2;y2;...;xN;yN] -> [x1 y1; x2 y2; ... ; xN yN]
    num_coords = length(coords_col);
    N = num_coords/2;
    M = [coords_col(1:2:num_coords-1), coords_col(2:2:num_coords)];
end

function coords_col = matrix_to_column(M)
% [x1 y1; x2 y2; ... ; xN yN] -> [x1;y1;...;xN;yN]
    N = size(M,1);
    coords_col = zeros(2*N,1);
    coords_col(1:2:end) = M(:,1);
    coords_col(2:2:end) = M(:,2);
end

%CONSTRAINTS 

function length_errors = length_error_func(vertex_coords, leg_params)
% Link length constraints:
% e_i = (xb - xa)^2 + (yb - ya)^2 - d_i^2 = 0  for each link i
    V = column_to_matrix(vertex_coords);   % N x 2
    L = leg_params.link_to_vertex_list;    % m x 2
    d = leg_params.link_lengths(:);        % m x 1

    m = size(L,1);
    length_errors = zeros(m,1);
    for i = 1:m
        a = L(i,1); b = L(i,2);
        dx = V(b,1) - V(a,1);
        dy = V(b,2) - V(a,2);
        length_errors(i) = dx*dx + dy*dy - d(i)^2;
    end
end

function coord_errors = fixed_coord_error_func(vertex_coords, leg_params, theta)
% Fixed vertex constraints:
%  vertex 1 lies on the crank circle centered at vertex 0 with radius crank_length,
%  vertex 2 is fixed at vertex_pos2
    V = column_to_matrix(vertex_coords);

    x1y1_target = leg_params.vertex_pos0 + leg_params.crank_length*[cos(theta); sin(theta)];
    x2y2_target = leg_params.vertex_pos2;

    coord_errors = [ ...
        V(1,1) - x1y1_target(1);   % x1 - x1_target
        V(1,2) - x1y1_target(2);   % y1 - y1_target
        V(2,1) - x2y2_target(1);   % x2 - x2_target
        V(2,2) - x2y2_target(2)];  % y2 - y2_target
end

function error_vec = linkage_error_func(vertex_coords, leg_params, theta)
% Combine distance + fixed coordinate constraints into one vector
    error_vec = [ ...
        length_error_func(vertex_coords, leg_params);
        fixed_coord_error_func(vertex_coords, leg_params, theta) ...
    ];
end

% WRAPPER 

function vertex_coords_root = compute_coords(vertex_coords_guess, leg_params, theta)
% Run Newton with numerical Jacobian (per Part 4 guidelines)
    fun = @(V) linkage_error_func(V, leg_params, theta);
    [root, ~, flag] = multi_newton_solver(fun, vertex_coords_guess, 1e-14, 1e-14, 200, 1);
    if flag ~= 1
        warning('Newton did not report success (flag=%d). Result may be off.', flag);
    end
    vertex_coords_root = root;
end

function dVdtheta = compute_velocities(vertex_coords, leg_params, theta)
    %get the jacobian of the error function F
    %create matrix M which is that jacobian plus I(4) and 0(4,10)
    %create vector A which is zeros(10,1) plus -rsin plus rcos plus
    %zeros(2,1)
    %instantaneous velocities dVdtheta = M\A
    f = @(V) length_error_func(V, leg_params);
    J = approximate_jacobian(f, vertex_coords);
    extra = [eye(4),zeros(4, 10)];

    r = leg_params.crank_length;
    M = [J;extra];
    A = [zeros(10,1); -r*sind(theta); r*cosd(theta); 0; 0];
    dVdtheta = M\A;
end

% SUMMARY OF WHAT THIS SCRIPT DOES
%
% 1 coded the Jansen linkage with 7 vertices and 10 links  
% 2 added extra constraints so vertex 1 moves on a crank circle and vertex 2 stays fixed  
% 3 wrote link length constraints to keep each pair of connected vertices the right distance  
% 4 wrote fixed coordinate constraints to lock vertex 1 on the circle and vertex 2 in place  
% 5 combined all constraints into one error function that should equal zero when satisfied  
%
% 6 built a multidimensional Newton solver to update guesses for the coordinates  
% 7 solver uses numerical Jacobian and stops when step size or residual is very small  
% 8 solver tracks iteration count and stores all guesses in a list  
%
% 9 set an initial guess for the 7 vertices based on the figure  
% 10 solver refined the guess to satisfy all constraints  
% 11 checked the residual norm to confirm solution is correct  
%
% 12 plotted the solved linkage with red dots for vertices and black lines for links  
% 13 drew the crank circle as a dashed outline for reference  
%
% 14 overall made functions for constraints, solver, and visualization  
% 15 result is a working solver that shows the leg configuration for any crank angle  
% 16 next step could be sweeping crank angle 0 to 2π to animate the linkage and track foot path  

