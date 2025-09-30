function animate_leg_motion(num_frames, num_rotations, fps)
% Jansen leg animation (no dotted overlay), shows only the live foot trail
% num_frames    : frames per rotation (default 240)  ↓ fewer → faster
% num_rotations : how many crank turns to show (default 3)
% fps           : playback rate (frames per second, default 30) ↑ bigger → faster

    if nargin < 1, num_frames    = 240; end
    if nargin < 2, num_rotations = 3;   end
    if nargin < 3, fps           = 120;  end
    pause_per_frame = 1 / fps;           % set to 0 for fastest possible

    % params + seed guess
    leg_params = build_leg_params();
    Vguess = [ ...
         0;  50;   -50;   0;   -50;  50; ...
       -100;   0;  -100; -50;   -50; -50;  -50; -100 ];

    % timeline over all rotations
    thetas = linspace(0, 2*pi*num_rotations, num_frames*num_rotations);

    % figure and static elements
    L  = leg_params.link_to_vertex_list;
    c0 = leg_params.vertex_pos0(:)';  R = leg_params.crank_length;

    figure; clf; hold on; axis equal; grid on;
    xlabel('x'); ylabel('y');
    title('Jansen leg animation');
    axis([-120 40 -120 40]);

    tt = linspace(0, 2*pi, 256);
    plot(c0(1)+R*cos(tt), c0(2)+R*sin(tt), 'k--', 'LineWidth', 1); % crank circle

    % graphics handles
    hLinks  = gobjects(size(L,1),1);
    for i = 1:size(L,1)
        hLinks(i) = plot([0 0],[0 0],'k-','LineWidth',2);
    end
    hVerts  = plot(0,0,'ro','MarkerFaceColor','r','MarkerSize',6);
    hCrankR = plot([0 0],[0 0],'k-','LineWidth',1.5);
    hArrow  = quiver(0,0,0,0,0,'Color',[0.1 0.4 1],'LineWidth',1.5,'MaxHeadSize',2);

    % live foot trail only (no dotted overlay)
    hTrail = plot(nan,nan,'b-','LineWidth',1.4);
    trail_x = nan(1, num_frames);
    trail_y = nan(1, num_frames);
    trail_k = 0;
    tip_ix = 7;

    for k = 1:numel(thetas)
        th = thetas(k);

        % reset trail at the start of each rotation
        if mod(th, 2*pi) < (2*pi/num_frames)
            trail_k = 0;
            set(hTrail, 'XData', nan, 'YData', nan);
        end

        % solve linkage + draw
        V  = compute_coords(Vguess, leg_params, th);
        Vm = column_to_matrix(V);

        for i = 1:size(L,1)
            a = L(i,1); b = L(i,2);
            set(hLinks(i), 'XData', [Vm(a,1) Vm(b,1)], 'YData', [Vm(a,2) Vm(b,2)]);
        end
        set(hVerts, 'XData', Vm(:,1), 'YData', Vm(:,2));
        set(hCrankR, 'XData', [c0(1) c0(1)+R*cos(th)], 'YData', [c0(2) c0(2)+R*sin(th)]);

        % velocity arrow at foot tip
        dVdth = compute_velocities(V, leg_params, th);
        x_tip = V(2*tip_ix-1); y_tip = V(2*tip_ix);
        vx    = dVdth(2*tip_ix-1); vy = dVdth(2*tip_ix);
        set(hArrow, 'XData', x_tip, 'YData', y_tip, 'UData', 0.4*vx, 'VData', 0.4*vy);

        % update live trail
        trail_k = trail_k + 1;
        if trail_k <= num_frames
            trail_x(trail_k) = x_tip; trail_y(trail_k) = y_tip;
            set(hTrail, 'XData', trail_x(1:trail_k), 'YData', trail_y(1:trail_k));
        end

        drawnow;
        pause(pause_per_frame);
    end
end

%% HELPER FUNCTIONS from day 7 and 9 (6 7)

function leg = build_leg_params()
    leg = struct();
    leg.num_vertices = 7;
    leg.num_linkages = 10;

    % link-to-vertex list (rows: [va vb]) 
    leg.link_to_vertex_list = [ ...
        1 3;  % 1
        3 4;  % 2
        2 3;  % 3
        2 4;  % 4
        4 5;  % 5
        2 6;  % 6
        1 6;  % 7
        5 6;  % 8
        5 7;  % 9
        6 7]; % 10

    % link lengths (same order as above)
    leg.link_lengths = [50.0; 55.8; 41.5; 40.1; 39.4; 39.3; 61.9; 36.7; 65.7; 49.0];

    % additional constraints
    leg.crank_length = 15.0;            % radius for vertex 1 around center
    leg.vertex_pos0  = [0; 0];          % crank center
    leg.vertex_pos2  = [-38.0; -7.8];   % fixed vertex 2
end

function Vmat = column_to_matrix(Vcol)
    Vmat = [Vcol(1:2:end), Vcol(2:2:end)];
end

function Vcol = matrix_to_column(Vmat)
    Vcol = Vmat.'; Vcol = Vcol(:);
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

function J = approximate_jacobian(FUN, X)
    h = 1e-6; n = numel(X);
    F0 = FUN(X); m = numel(F0);
    J = zeros(m,n);
    for i = 1:n
        e = zeros(n,1); e(i) = h/2;
        J(:,i) = (FUN(X+e) - FUN(X-e))/h;
    end
end

function [root, it, flag] = multi_newton_solver(F, x0, Ath, Bth, maxit, numdiff)
    x = x0; it = 0; flag = 0;
    r = F(x);

    while norm(r) > Bth && it < maxit
        if numdiff == 2
            try
                [r,J] = F(x); %ok
            catch
                r = F(x);
                J = approximate_jacobian(F,x);
            end
        else
            J = approximate_jacobian(F,x);
        end

        if abs(det(J*J')) < eps, flag = -2; return, end

        s = -J\r;
        xnew = x + s;

        if norm(xnew-x) < Ath, x = xnew; flag = 1; break, end

        x = xnew; r = F(x); it = it+1;
    end

    if norm(r) <= Bth, flag = 1; end
    root = x;
end

function [Vroot, flag, resnorm] = compute_coords(Vguess, leg, theta)
    fun = @(V) linkage_error_func(V, leg, theta);

    % slightly relaxed thresholds + more iters for stubborn angles
    [Vroot, ~, flag] = multi_newton_solver(fun, Vguess, 1e-12, 1e-12, 300, 1);

    resnorm = norm(fun(Vroot));
    if flag ~= 1 && resnorm < 1e-8
        flag = 1; % accept near-zero residuals without printing
    end
end

function dVdtheta = compute_velocities(V, leg, theta)
% theta-derivatives of all 14 coordinates via linear system 
    Jlen = approximate_jacobian(@(X) length_error_func(X,leg), V);
    r = leg.crank_length;
    dx1 = -r*sin(theta); dy1 = r*cos(theta);
    dx2 = 0; dy2 = 0;
    M = [eye(4), zeros(4,10); Jlen];
    B = [dx1; dy1; dx2; dy2; zeros(10,1)];
    dVdtheta = M\B;
end


% SUMMARY OF WHAT THIS CODE DOES (6 7)
% 1) built leg parameters for the Jansen linkage from the handout
%    link to vertex list
%    link lengths
%    fixed crank center and vertex 2
%    crank radius for vertex 1

% 2) accepted runtime controls from function inputs
%    num_frames frames per rotation   fewer → faster
%    num_rotations how many full turns to show
%    fps playback rate in frames per second   bigger → faster
%    pause_per_frame = 1/fps sets the delay between frames

% 3) created a timeline of crank angles
%    thetas spans 0 to 2π times num_rotations with num_frames per rotation

% 4) prepared the figure once
%    fixed axes and grid
%    drew the static crank circle for reference
%    preallocated graphics handles for links vertices crank arm velocity arrow and live trail
%    preallocation avoids recreating graphics each frame and keeps animation smooth

% 5) per frame logic
%    a) reset the live foot trail at the start of each rotation so one clean loop shows
%    b) solved the linkage constraints at angle th using compute_coords
%       compute_coords wraps multidimensional Newton with a numerical Jacobian
%       termination uses small step size threshold and small residual threshold
%       jacobian safeguard avoids singular or near singular systems
%    c) updated link segments using the link to vertex list
%    d) updated vertex markers and the current crank radius line
%    e) computed dV/dθ using compute_velocities
%       assembled matrix M = [I(4) 0 ; J_length] and right hand side B for the fixed coords
%       solved M * dVdθ = B to get coordinate derivatives with respect to θ
%    f) drew a velocity arrow at the foot tip   scaled by 0.4 for visibility
%    g) extended the live trail with the current foot tip position
%    h) advanced the frame with drawnow and a pause of 1/fps seconds

% 6) helper functions summary
%    build_leg_params packs all constants for the linkage
%    column_to_matrix and matrix_to_column convert between [x1;y1;...;xN;yN] and [x y] rows
%    length_error_func enforces squared distance of each link equals d_i^2
%    fixed_coord_error_func enforces vertex 1 on crank circle and vertex 2 fixed in space
%    linkage_error_func concatenates the two error vectors for Newton
%    approximate_jacobian builds a centered finite difference Jacobian
%    multi_newton_solver runs Newton with numerical J and simple safeguards
%    compute_coords calls Newton and returns the vertex vector at a given θ
%    compute_velocities builds and solves the linear system for dV/dθ

% 7) how to change speed or smoothness
%    increase fps to speed playback   example animate_leg_motion(240,3,120)
%    decrease fps to slow playback   example animate_leg_motion(240,3,30)
%    reduce num_frames to compute fewer Newton solves per rotation and run faster
%    increase num_frames to make motion smoother at the cost of more solves

% 8) how to change arrow look
%    change the 0.4 scale factor in the quiver update for longer or shorter arrows
%    change color or linewidth in the quiver call if desired

% 9) typical troubleshooting
%    if Newton prints warnings occasionally but the animation looks stable   tolerances are just strict
%    if a frame looks jumpy   lower fps or increase num_frames
%    if the arrow looks too noisy   increase num_frames or reduce the quiver scale

