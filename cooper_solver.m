function [root, it, flag, glist] = cooper_solver()
    % test_numerical_jacobian()
    proj_targ_dist = @(thetat) projectile_traj(thetat(1),thetat(2)) - target_traj(thetat(2));
    [root, it, flag, glist] = multi_newton_solver(proj_targ_dist, 5*ones(2, 1), 1e-14, 1e-14, 200, 1);
    run_simulation(root(1),root(2))
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


function [F_val, J] = test_func01(x)
    f1 = x(1)^2 + x(2)^2 - 6 - x(3)^5;
    f2 = x(1)*x(3) + x(2) - 12;
    f3 = sin(x(1) + x(2) + x(3));
    df1x = 2*x(1);
    df1y = 2*x(2);
    df1z = -5*x(3)^4;
    df2x = x(3);
    df2y = 1;
    df2z = x(1);
    df3x = cos(x(1) + x(2) + x(3));
    df3y = cos(x(1) + x(2) + x(3));
    df3z = cos(x(1) + x(2) + x(3));
    F_val = [f1;f2;f3];
    J = [df1x, df1y, df1z; df2x, df2y, df2z; df3x, df3y, df3z];
end

function [f_out,dfdx] = test_func02(X)
    x1 = X(1);
    x2 = X(2);
    x3 = X(3);
    y1 = 3*x1^2 + .5*x2^2 + 7*x3^2-100;
    y2 = 9*x1-2*x2+6*x3;
    f_out = [y1;y2];
    dfdx = [6*x1,1*x2,14*x3;9,-2,6];
end

function V_p = projectile_traj(theta,t)
    g = 2.3; %gravity in m/sˆ2
    v0 = 14; %initial speed in m/s
    px0 = 2; %initial x position
    py0 = 4; %initial y position
    %compute position vector
    V_p = [v0*cos(theta)*t+px0; -.5*g*t.^2+v0*sin(theta)*t+py0];
end

function V_t = target_traj(t)
    a1 = 7; %x amplitude in meters
    b1 = 9; %y amplitude meters
    omega1 = 3; %frequency in rad/sec
    phi1 = -pi/7; %phase shift in radians
    a2 = 2; %x amplitude in meters
    b2 = .7; %y amplitude meters
    omega2 = 5; %frequency in rad/sec
    phi2 = 1.5*pi; %phase shift in radians
    x0 = 28; %x offset in meters
    y0 = 21; %y offset in meters
    %compute position vector
    V_t = [a1*cos(omega1*t+phi1)+a2*cos(omega2*t+phi2)+x0;...
    b1*sin(omega1*t+phi1)+b2*sin(omega2*t+phi2)+y0];
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

function test_numerical_jacobian()
    %number of tests to perform
    num_tests = 100;
    %iterate num_tests times
    for n = 1:num_tests
        %generate a randomized input and output dimension
        input_dim = randi([1,15]);
        output_dim = randi([1,15]);
        %generate a input_dim x input_dim x output_dim matrix stack A
        A = randn(input_dim,input_dim,output_dim);
        %generate a matrix, B of dimension output_dim x input_dim
        B = randn(output_dim,input_dim);
        %generate a column vector, C of height output_dim
        C = randn(output_dim,1);
        %create a new test function
        %this is essentially a random second-order (quadratic) function
        %with input dimension input_dim and output dimension output_dim
        test_fun = @(X) jacobian_test_function(X,A,B,C);
        X_guess = randn(input_dim,1);
        %evaluate numerical Jacobian of test_fun
        %use whatever your function name was here!
        J_numerical = approximate_jacobian(test_fun,X_guess);
        %compute the analytical jacobian of jacobian_test_function
        J_analytical = B;
        for n = 1:output_dim
            J_analytical(n,:)=J_analytical(n,:)+X_guess'*A(:,:,n);
            J_analytical(n,:)=J_analytical(n,:)+X_guess'*A(:,:,n)';
        end
        %compare with Jacobian of A
        largest_error = max(max(abs(J_numerical-J_analytical)));
        %if J is not close to A, print fail.
        if largest_error>1e-7
        disp('fail!');
        end
    end
end
%computes a quadratic function on input X
function f_val = jacobian_test_function(X,A,B,C)
    output_length = length(C);
    f_val = B*X+C;
    for n = 1:output_length
        f_val(n)=f_val(n)+(X'*A(:,:,n)*X);
    end
end

function run_simulation(theta,t_c)
    %create the plot window, set the axes size, and add labels
    hold on;
    axis equal; axis square;
    axis([0,50,0,50])
    xlabel('x (m)')
    ylabel('y (m)')
    title('Simulation of Projectile Shot at Target')
    %initialize plots for the projectile/target and their paths
    traj_line_proj = plot(0,0,'g--','linewidth',2);
    traj_line_targ = plot(0,0,'k--','linewidth',2);
    t_plot = plot(0,0,'bo','markerfacecolor','b','markersize',8);
    p_plot = plot(0,0,'ro','markerfacecolor','r','markersize',8);
    %position lists
    %used for plotting paths of projectile/target
    V_list_proj = [];
    V_list_targ = [];
    %iterate through time until a little after the collision occurs
    for t = 0:.005:t_c+1.5
        %set time so that things freeze once collision occurs
        t_input = min(t,t_c);
        %compute position of projectile and target
        V_p = projectile_traj(theta,t_input);
        V_t = target_traj(t_input);
        %update the position lists
        V_list_proj(:,end+1) = V_p;
        V_list_targ(:,end+1) = V_t;
        %index used for tail of target path
        i = max(1,size(V_list_targ,2)-300);
        %update plots
        set(t_plot,'xdata',V_t(1),'ydata',V_t(2));
        set(p_plot,'xdata',V_p(1),'ydata',V_p(2));
        set(traj_line_proj,'xdata',V_list_proj(1,:),'ydata',V_list_proj(2,:));
        set(traj_line_targ,'xdata',V_list_targ(1,i:end),'ydata',V_list_targ(2,i:end));
        %show updated plots
        drawnow;
    end
end

