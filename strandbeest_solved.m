function strandbeest_solved()
    %initialize leg_params structure
    leg_params = struct();
    %number of vertices in linkage
    leg_params.num_vertices = 7;
    %number of links in linkage
    leg_params.num_linkages = 10;
    %matrix relating links to vertices
    leg_params.link_to_vertex_list = ...
    [ 1, 3;... %link 1 adjacency
    3, 4;... %link 2 adjacency
    2, 3;... %link 3 adjacency
    2, 4;... %link 4 adjacency
    4, 5;... %link 5 adjacency
    2, 6;... %link 6 adjacency
    1, 6;... %link 7 adjacency
    5, 6;... %link 8 adjacency
    5, 7;... %link 9 adjacency
    6, 7 ... %link 10 adjacency
    ];

    %list of lengths for each link
    %in the leg mechanism
    leg_params.link_lengths = ...
    [ 50.0,... %link 1 length
    55.8,... %link 2 length
    41.5,... %link 3 length
    40.1,... %link 4 length
    39.4,... %link 5 length
    39.3,... %link 6 length
    61.9,... %link 7 length
    36.7,... %link 8 length
    65.7,... %link 9 length
    49.0 ... %link 10 length
    ];

    %length of crank shaft
    leg_params.crank_length = 15.0;
    %fixed position coords of vertex 0
    leg_params.vertex_pos0 = [0;0];
    %fixed position coords of vertex 2
    leg_params.vertex_pos2 = [-38.0;-7.8];


    %column vector of initial guesses
    %for each vertex location.
    %in form: [x1;y1;x2;y2;...;xn;yn]
    vertex_coords_guess = [...
    [ 0; 50];... %vertex 1 guess
    [ -50; 0];... %vertex 2 guess
    [ -50; 50];... %vertex 3 guess
    [-100; 0];... %vertex 4 guess
    [-100; -50];... %vertex 5 guess
    [ -50; -50];... %vertex 6 guess
    [ -50; -100]... %vertex 7 guess
    ];

    theta = 0;
    root = compute_coords(vertex_coords_guess, leg_params, theta);
    leg_drawing = initialize_leg_drawing(leg_params);
end

%Error function that encodes the link length constraints
%INPUTS:
%vertex_coords: a column vector containing the (x,y) coordinates
%leg_params: a struct containing the parameters that describe the linkage
% importantly, leg_params.link_lengths is a list of linakge lengths
% and leg_params.link_to_vertex_list is a two column matrix where
% leg_params.link_to_vertex_list(i,1) and
% leg_params.link_to_vertex_list(i,2) are the pair of vertices connected
% by the ith link in the mechanism
%OUTPUTS:
%length_errors: a column vector describing the current distance error of the ith
% link specifically, length_errors(i) = (xb-xa)ˆ2 + (yb-ya)ˆ2 - d_iˆ2
% where (xa,ya) and (xb,yb) are the coordinates of the vertices that
% are connected by the ith link, and d_i is the length of the ith link
function length_errors = link_length_error_func(vertex_coords, leg_params)
    linklens = leg_params.link_lengths;
    linkverts = leg_params.link_to_vertex_list;
    M = column_to_matrix(vertex_coords);
    length_errors = zeros(length(linklens),1);
    for i = 1:length(linklens)
        grab = linkverts(i,:);
        store1 = M(grab(1),:);
        store2 = M(grab(2),:);
        xa = store1(1);
        xb = store2(1);
        ya = store1(2);
        yb = store2(2);
        d = linklens(i);
        err = (xb-xa)^2 + (yb-ya)^2 - d^2;
        length_errors(i) = err;
    end
end

function coord_errors = fixed_coord_error_func(vertex_coords, leg_params, theta)
    r = leg_params.crank_length;
    coords0 = leg_params.vertex_pos0;
    coords2 = leg_params.vertex_pos2;
    xbar1 = coords0(1) + r*cosd(theta);
    ybar1 = coords0(2) + r*sind(theta);
    xbar2 = coords2(1);
    ybar2 = coords2(2);
    coord_errors = zeros(4,1);
    coord_errors(1) = vertex_coords(1)-xbar1;
    coord_errors(2) = vertex_coords(2)-ybar1;
    coord_errors(3) = vertex_coords(3)-xbar2;
    coord_errors(4) = vertex_coords(4)-ybar2;
end

function error_vec = linkage_error_func(vertex_coords, leg_params, theta)
    distance_errors = link_length_error_func(vertex_coords, leg_params);
    coord_errors = fixed_coord_error_func(vertex_coords, leg_params, theta);
    error_vec = [distance_errors;coord_errors];
end

function root = compute_coords(vertex_coords_guess, leg_params, theta)
    f = @(V) linkage_error_func(V, leg_params, theta);
    [root, it, flag, glist] = multi_newton_solver(f,  vertex_coords_guess, 1e-14, 1e-14, 200, 1);
end

function coords_out = column_to_matrix(coords_in)
    num_coords = length(coords_in);
    coords_out = [coords_in(1:2:(num_coords-1)),coords_in(2:2:num_coords)];
end

function coords_out = matrix_to_column(coords_in)
    num_coords = 2*size(coords_in,1);
    coords_out = zeros(num_coords,1);
    coords_out(1:2:(num_coords-1)) = coords_in(:,1);
    coords_out(2:2:num_coords) = coords_in(:,2);
end


function [root, it, flag, glist] = multi_newton_solver(FUN_both, X0, Athresh, Bthresh, maxit, numdiff)
    % 1 = numerical, 2 = analytical
    % fast convergence near     root
    % using slope to jump closer
    glist = [];                             % step 1: start xn list
    root = X0; it = 0; flag = 0;
    glist(:, end+1) = root;                    % step 1: save first guess

    if numdiff == 2
        try
            [FX, J] = FUN_both(root);  % Try to get both outputs (analytical)
        catch
            FX = FUN_both(root);       % Fallback if FUN_both only returns one output
            J = approximate_jacobian(FUN_both, root);
        end
    else
        FX = FUN_both(root);           % Numerical mode: always get 1 output
        J = approximate_jacobian(FUN_both, root);
    end

    if numdiff == 1
        J = approximate_jacobian(FUN_both, X0);
    end
    while norm(FX) > Bthresh && it < maxit
        if det(J*J') == 0
            flag = -2; return               % -2 = zero derivative
        end
        X_new = root - J\FX;              % updating step
        glist(:, end+1) = X_new;                % step 1: save iterate (kept as is)
        if abs(X_new - root) < Athresh
            root = X_new; flag = 1; return
        end
        root = X_new;
        if numdiff == 2
            try
                [FX, J] = FUN_both(root);  % Try to get both outputs (analytical)
            catch
                FX = FUN_both(root);       % Fallback if FUN_both only returns one output
                J = approximate_jacobian(FUN_both, root);
            end
        else
            FX = FUN_both(root);           % Numerical mode: always get 1 output
            J = approximate_jacobian(FUN_both, root);
        end
        it = it + 1;
    end
    flag = 1;
end

function J = approximate_jacobian(FUN,X)
    h = 1e-6;
    J = [];
    varnum = length(X);
    for i = 1:varnum
        ei = zeros(varnum, 1);
        ei(i) = h/2;
        J(:,end+1) = (FUN(X+ei)-FUN(X-ei))./h;
    end

end

function leg_drawing = initialize_leg_drawing(leg_params)
    leg_drawing = struct();
    leg_drawing.linkages = cell(leg_params.num_linkages,1);
    for linkage_index = 1:leg_params.num_linkages
        leg_drawing.linkages{linkage_index} = line([0,0],[0,0],'color','k','linewidth',2);
    end
    leg_drawing.crank = line([0,0],[0,0],'color','k','linewidth',1.5);
    leg_drawing.vertices = cell(leg_params.num_vertices,1);
    for vertex_index = 1:leg_params.num_vertices
        leg_drawing.vertices{vertex_index} = line([0],[0],'marker',...
        'o','markerfacecolor','r','markeredgecolor','r','markersize',8);
    end
end
