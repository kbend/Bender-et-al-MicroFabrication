% --- Step 1: Import Data ---
data = readmatrix('name of data.xlsx');

% --- Step 2: Download circistat-matlab and add as path link: https://github.com/circstat/circstat-matlab ---

% --- Step 3: Convert from degrees to radians ---
control = deg2rad(data(:,1));   % first column = control
exp     = deg2rad(data(:,2));   % second column = experimental

% Remove NaNs from each group
control = control(~isnan(control));
exp     = exp(~isnan(exp));

% --- Step 4: Combine data and define group labels ---
angles = [control; exp];                      % all data in one vector
group  = [ones(size(control)); 2*ones(size(exp))];  % 1 = control, 2 = experimental

% --- Step 5: Run Watson–Williams test ---
[p, table] = circ_wwtest(angles, group);

% --- Step 6: Display results ---
disp('--- Watson–Williams Test Results ---');
disp(table);

fprintf('\nP-value = %.5f\n', p);
