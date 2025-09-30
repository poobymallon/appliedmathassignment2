function day7_linkage_demo()
% Solve the Jansen-leg linkage for a given crank angle and plot the result
% using link segments from the link-to-vertex list (matches handout)

    %parameters (from handout) 
    leg_params = build_leg_params();

    % choose a crank angle (radians)
    theta = 0.35;

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

%% day 9 stuff
function dVdtheta = compute_velocities(vertex_coords, leg_params, theta)
% theta-derivatives of all 14 coordinates using the linear system from the notes

    % 1) Jacobian of the 10 link-length errors wrt V  (10x14)
    Jlen = approximate_jacobian(@(V) length_error_func(V,leg_params), vertex_coords);

    % 2) derivatives of fixed constraints
    r = leg_params.crank_length;
    dx1_dtheta = -r * sin(theta);
    dy1_dtheta =  r * cos(theta);
    dx2_dtheta = 0;
    dy2_dtheta = 0;

    % 3) build M (14x14) and B (14x1)
    M = [ eye(4) , zeros(4,10) ;
          Jlen                 ];
    B = [dx1_dtheta ; dy1_dtheta ; dx2_dtheta ; dy2_dtheta ; zeros(10,1)];

    % 4) solve
    dVdtheta = M \ B;
end

function dVdtheta_num = numerical_dVdtheta(theta, vertex_guess_coords, leg_params)
% central difference on the solved configuration V(theta)
    h  = 1e-4;  % radians
    Vp = compute_coords(vertex_guess_coords, leg_params, theta + h);
    Vm = compute_coords(vertex_guess_coords, leg_params, theta - h);
    dVdtheta_num = (Vp - Vm) / (2*h);
end

function draw_tip_velocity_arrow(ax, V, dVdtheta, scale)
% arrow at vertex 7 showing direction of motion wrt theta
    if nargin < 4, scale = 1; end
    tip_ix = 7;
    x  = V(2*tip_ix-1);  y  = V(2*tip_ix);
    vx = dVdtheta(2*tip_ix-1); vy = dVdtheta(2*tip_ix);
    quiver(ax, x, y, scale*vx, scale*vy, 0, ...
           'Color',[0.1 0.4 1], 'LineWidth',1.5, 'MaxHeadSize',2);
end

function day9_compare_velocity_curves()
% compares analytic vs numerical derivatives of the foot tip (vertex 7)

    leg_params = build_leg_params();
    tip_ix = 7;
    Vguess = [ ...
         0;  50;   -50;   0;   -50;  50; ...
       -100;   0;  -100; -50;   -50; -50;  -50; -100 ];

    thetas = linspace(0, 2*pi, 200);
    dx_ana = zeros(size(thetas));
    dy_ana = zeros(size(thetas));
    dx_num = zeros(size(thetas));
    dy_num = zeros(size(thetas));

    for k = 1:numel(thetas)
        th = thetas(k);

        V = compute_coords(Vguess, leg_params, th);
        dVdth     = compute_velocities(V, leg_params, th);
        dVdth_num = numerical_dVdtheta(th, Vguess, leg_params);

        dx_ana(k) = dVdth(2*tip_ix-1);
        dy_ana(k) = dVdth(2*tip_ix);
        dx_num(k) = dVdth_num(2*tip_ix-1);
        dy_num(k) = dVdth_num(2*tip_ix);
    end

    figure; hold on; grid on
    plot(thetas, dx_ana, '-',  'LineWidth',1.6)
    plot(thetas, dx_num, '--', 'LineWidth',1.6)
    xlabel('\theta (rad)'); ylabel('d x_{tip} / d\theta')
    title('dx_{tip}/d\theta vs \theta  analytic (solid)  numerical (dashed)')
    legend('analytic','numerical','Location','best')

    figure; hold on; grid on
    plot(thetas, dy_ana, '-',  'LineWidth',1.6)
    plot(thetas, dy_num, '--', 'LineWidth',1.6)
    xlabel('\theta (rad)'); ylabel('d y_{tip} / d\theta')
    title('dy_{tip}/d\theta vs \theta  analytic (solid)  numerical (dashed)')
    legend('analytic','numerical','Location','best')
end

function animate_leg_motion(num_frames)
% animates one full crank revolution of the Jansen leg with a velocity arrow
% num_frames: optional, frames over [0,2pi]. default 240.

    if nargin < 1, num_frames = 240; end

    % params and a reasonable seed guess (same layout you used)
    leg_params = build_leg_params();
    Vguess = [ ...
         0;  50;   -50;   0;   -50;  50; ...
       -100;   0;  -100; -50;   -50; -50;  -50; -100 ];

    thetas = linspace(0, 2*pi, num_frames);
    L = leg_params.link_to_vertex_list;              % link list
    c = leg_params.vertex_pos0(:)'; R = leg_params.crank_length; % crank

    % figure + static crank circle
    figure; clf; hold on; axis equal; grid on;
    xlabel('x'); ylabel('y');
    title('Jansen leg animation');
    axis([-120 40 -120 40]);

    t = linspace(0, 2*pi, 256);
    plot(c(1)+R*cos(t), c(2)+R*sin(t), 'k--', 'LineWidth', 1); % crank circle

    % pre-create graphics handles (faster than cla each frame)
    hLinks  = gobjects(size(L,1),1);
    for i = 1:size(L,1)
        hLinks(i) = plot([0 0],[0 0],'k-','LineWidth',2);
    end
    hVerts  = plot(0,0,'ro','MarkerFaceColor','r','MarkerSize',6);
    hCrankR = plot([0 0],[0 0],'k-','LineWidth',1.5);
    hArrow  = quiver(0,0,0,0,0,'Color',[0.1 0.4 1],'LineWidth',1.5,'MaxHeadSize',2);
    hFoot   = plot(nan,nan,'b:','LineWidth',1);       % foot path trail

    foot_x = nan(1,num_frames);
    foot_y = nan(1,num_frames);

    for k = 1:num_frames
        th = thetas(k);

        % solve linkage at this theta
        V = compute_coords(Vguess, leg_params, th);
        Vm = column_to_matrix(V);

        % update links
        for i = 1:size(L,1)
            a = L(i,1); b = L(i,2);
            set(hLinks(i), 'XData', [Vm(a,1) Vm(b,1)], 'YData', [Vm(a,2) Vm(b,2)]);
        end

        % update vertices
        set(hVerts, 'XData', Vm(:,1), 'YData', Vm(:,2));

        % update crank radius line
        set(hCrankR, 'XData', [c(1) c(1)+R*cos(th)], 'YData', [c(2) c(2)+R*sin(th)]);

        % update velocity arrow at the tip (vertex 7)
        dVdth = compute_velocities(V, leg_params, th);
        tip_ix = 7;
        x  = V(2*tip_ix-1);   y  = V(2*tip_ix);
        vx = dVdth(2*tip_ix-1); vy = dVdth(2*tip_ix);
        scale = 0.4;  % tweak for visual length
        set(hArrow, 'XData', x, 'YData', y, 'UData', scale*vx, 'VData', scale*vy);

        % foot path trail
        foot_x(k) = x; foot_y(k) = y;
        set(hFoot, 'XData', foot_x, 'YData', foot_y);

        drawnow;
    end
end

% SUMMARY OF WHAT THIS FILE DOES (Day 7 + Day 9) (6 7)
%
% 1) Build leg geometry with build_leg_params
%    - define link-to-vertex list (which vertices each bar connects)
%    - store link lengths in same order as list
%    - fix crank center (vertex 0) and vertex 2 position
%    - specify crank radius for vertex 1
%
% 2) Define constraint functions
%    - length_error_func enforces each bar keeps its target length
%    - fixed_coord_error_func locks vertex 1 on the crank circle
%      and vertex 2 at its fixed coordinates
%    - linkage_error_func combines both into one residual vector
%
% 3) Solve the nonlinear system with Newton
%    - multi_newton_solver iterates until constraints are satisfied
%    - approximate_jacobian provides Jacobian by finite differences
%    - compute_coords wraps this to solve linkage for given crank angle
%
% 4) Day 7 linkage demo
%    - pick one crank angle
%    - solve positions of all 7 vertices
%    - check residual size to confirm validity
%    - plot links, joints, crank circle, and current crank radius
%
% 5) Day 9 velocity analysis
%    - compute_velocities builds a linear system to solve for
%      dV/dθ of all coordinates (analytic derivative)
%    - numerical_dVdtheta estimates same derivative by finite difference
%    - draw_tip_velocity_arrow shows the velocity vector of foot tip
%
% 6) Compare analytic vs numerical curves
%    - day9_compare_velocity_curves loops over θ from 0→2π
%    - computes dx_tip/dθ and dy_tip/dθ both ways
%    - plots results (analytic solid, numerical dashed)
%
% 7) Animate leg motion
%    - animate_leg_motion runs through multiple θ values
%    - updates link positions and vertex markers
%    - shows crank turning
%    - overlays velocity arrow at foot tip
%    - adds dotted trail of foot path
% ended up just making a different file for the compare curve
%and animation because its easier that way for me, lowkey just got lazy(6
%7)