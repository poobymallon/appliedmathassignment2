function compare_tip_velocity_curves()
% Compare analytic vs numerical derivatives of the foot tip (vertex 7)
% Produces dx_tip/dθ and dy_tip/dθ plots, analytic (solid) vs numerical (dashed)

    % Build leg geometry
    leg_params = build_leg_params();
    tip_ix = 7;

    % Initial guess for the linkage
    Vguess = [ ...
         0;  50;   -50;   0;   -50;  50; ...
       -100;   0;  -100; -50;   -50; -50;  -50; -100 ];

    % θ values to test
    Ntheta = 150;                  % fewer samples → faster
    thetas = linspace(0, 2*pi, Ntheta);

    dx_ana = zeros(size(thetas));
    dy_ana = zeros(size(thetas));
    dx_num = zeros(size(thetas));
    dy_num = zeros(size(thetas));

    % warm-start with first theta
    Vprev = compute_coords(Vguess, leg_params, thetas(1));

    for k = 1:numel(thetas)
        th = thetas(k);

        % use previous converged solution as guess
        V = compute_coords(Vguess, leg_params, th, Vprev);
        Vprev = V;

        % analytic derivative
        dVdth = compute_velocities(V, leg_params, th);

        % numerical derivative with warm-starts
        h = 1e-3;   % step size
        Vp = compute_coords(Vguess, leg_params, th + h, V);
        Vm = compute_coords(Vguess, leg_params, th - h, V);
        dVdth_num = (Vp - Vm)/(2*h);

        % extract tip vertex components
        dx_ana(k) = dVdth(2*tip_ix-1);
        dy_ana(k) = dVdth(2*tip_ix);
        dx_num(k) = dVdth_num(2*tip_ix-1);
        dy_num(k) = dVdth_num(2*tip_ix);
    end

    % Plot dx/dθ
    figure; hold on; grid on
    plot(thetas, dx_ana, '-',  'LineWidth',1.6)
    plot(thetas, dx_num, '--', 'LineWidth',1.6)
    xlabel('\theta (rad)'); ylabel('d x_{tip} / d\theta')
    title('Foot-tip d x / d\theta (analytic solid, numerical dashed)')
    legend('analytic','numerical','Location','best')

    % Plot dy/dθ
    figure; hold on; grid on
    plot(thetas, dy_ana, '-',  'LineWidth',1.6)
    plot(thetas, dy_num, '--', 'LineWidth',1.6)
    xlabel('\theta (rad)'); ylabel('d y_{tip} / d\theta')
    title('Foot-tip d y / d\theta (analytic solid, numerical dashed)')
    legend('analytic','numerical','Location','best')
end

%% Helper functions 

function Vroot = compute_coords(Vguess, leg, theta, x0)
    % Solve linkage at angle theta using Newton
    if nargin < 4, x0 = Vguess; end
    fun = @(V) linkage_error_func(V, leg, theta);
    [Vroot,~,flag] = multi_newton_solver(fun, x0, 1e-12, 1e-12, 400, 1);
    if flag~=1
        % fallback to template guess if warm-start failed
        [Vroot,~,flag] = multi_newton_solver(fun, Vguess, 1e-12, 1e-12, 400, 1);
        % comment out next line if don’t want  see warnings
        % if flag~=1, warning('Newton did not fully converge (flag=%d)',flag); end
    end
end

function dVdtheta = compute_velocities(V, leg, theta)
    Jlen = approximate_jacobian(@(X) length_error_func(X,leg), V);
    r = leg.crank_length;
    dx1 = -r*sin(theta); dy1 = r*cos(theta);
    dx2 = 0; dy2 = 0;
    M = [eye(4), zeros(4,10); Jlen];
    B = [dx1; dy1; dx2; dy2; zeros(10,1)];
    dVdtheta = M\B;
end

function err = linkage_error_func(V, leg, theta)
    err = [length_error_func(V, leg); fixed_coord_error_func(V, leg, theta)];
end

function len_err = length_error_func(Vcol, leg)
    V = column_to_matrix(Vcol);
    L = leg.link_to_vertex_list; d = leg.link_lengths(:);
    m = size(L,1); len_err = zeros(m,1);
    for i = 1:m
        a = L(i,1); b = L(i,2);
        dx = V(b,1)-V(a,1); dy = V(b,2)-V(a,2);
        len_err(i) = dx*dx + dy*dy - d(i)^2;
    end
end

function ce = fixed_coord_error_func(Vcol, leg, theta)
    V = column_to_matrix(Vcol);
    x1y1_t = leg.vertex_pos0 + leg.crank_length*[cos(theta); sin(theta)];
    x2y2_t = leg.vertex_pos2;
    ce = [V(1,1)-x1y1_t(1); V(1,2)-x1y1_t(2); V(2,1)-x2y2_t(1); V(2,2)-x2y2_t(2)];
end

function Vmat = column_to_matrix(Vcol)
    N = numel(Vcol)/2;
    Vmat = [Vcol(1:2:end), Vcol(2:2:end)];
    if size(Vmat,1)~=N, Vmat = reshape(Vcol,2,[])'; end
end

function J = approximate_jacobian(FUN, X)
    h = 1e-6; n = numel(X); J = zeros(numel(FUN(X)),n);
    for i = 1:n
        e = zeros(n,1); e(i)=h/2;
        J(:,i) = (FUN(X+e) - FUN(X-e))/h;
    end
end

function [root, it, flag] = multi_newton_solver(F, x0, Ath, Bth, maxit, numdiff)
    x = x0; it = 0; flag = 0; r = F(x);
    while norm(r) > Bth && it < maxit
        J = approximate_jacobian(F,x);
        if abs(det(J*J')) < eps, flag = -2; return, end
        xnew = x - J\r;
        if norm(xnew-x) < Ath, x = xnew; flag = 1; break, end
        x = xnew; r = F(x); it = it+1;
    end
    if norm(r) <= Bth, flag = 1; end
    root = x;
end

function leg = build_leg_params()
    leg.num_vertices = 7; leg.num_linkages = 10;
    leg.link_to_vertex_list = [1 3; 3 4; 2 3; 2 4; 4 5; 2 6; 1 6; 5 6; 5 7; 6 7];
    leg.link_lengths = [50.0;55.8;41.5;40.1;39.4;39.3;61.9;36.7;65.7;49.0];
    leg.crank_length = 15.0; leg.vertex_pos0=[0;0]; leg.vertex_pos2=[-38.0;-7.8];
end

%SUMMARY OF WHAT CODE DOES (6 7)
% 1) set up leg geometry with build_leg_params
%    defined link to vertex list
%    stored link lengths
%    fixed the crank center and vertex 2
%    specified crank radius for vertex 1

% 2) prepared the problem
%    tip_ix = 7 marks the foot tip vertex
%    Vguess defined the initial coordinate layout for Newton solver
%    created a range of θ values (thetas) to sample from 0 to 2π
%    allocated arrays to hold analytic and numerical velocity components

% 3) warm start method
%    solved first θ with Vguess and stored solution in Vprev
%    for each next θ reused Vprev as the starting guess
%    warm starts improve speed and convergence stability

% 4) loop over θ values
%    a) solved the linkage configuration at θ using compute_coords
%       compute_coords runs Newton solver with error function
%       if warm start fails falls back to template guess
%    b) computed analytic derivatives with compute_velocities
%       assembled M and B system from jacobian of link length constraints
%       solved for dV/dθ
%    c) computed numerical derivatives
%       central difference with step h = 1e-3
%       solved linkage at θ+h and θ-h then differenced and divided by 2h
%    d) extracted dx/dθ and dy/dθ for vertex 7
%       saved both analytic and numerical results into arrays

% 5) plotted results
%    first figure shows dx_tip/dθ vs θ
%       solid line = analytic   dashed line = numerical
%       legend differentiates the two
%    second figure shows dy_tip/dθ vs θ
%       same analytic vs numerical overlay

% 6) helper function roles
%    compute_coords runs Newton with linkage constraints at given θ
%    compute_velocities assembles and solves linear system for analytic dV/dθ
%    linkage_error_func bundles length and fixed coordinate constraints
%    length_error_func enforces squared distance of links equal to lengths^2
%    fixed_coord_error_func enforces vertex 1 on crank circle and vertex 2 fixed
%    column_to_matrix converts [x1;y1;...;xN;yN] into N×2 [x y] rows
%    approximate_jacobian computes centered difference Jacobian
%    multi_newton_solver implements Newton iteration with tolerance and safeguards
%    build_leg_params stores linkage constants

% 7) tuning controls
%    reduce Ntheta for faster runtime with fewer sample points
%    adjust h in numerical derivative for accuracy vs noise
%    relax thresholds in multi_newton_solver if warnings appear too often

% 8) interpretation
%    plots should show analytic and numerical curves nearly overlapping
%    small deviations are expected near singularities or crank extremes
%    runtime warnings of Newton not fully converging are normal but can be reduced


