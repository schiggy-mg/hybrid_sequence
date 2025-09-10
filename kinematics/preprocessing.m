% creat mat files for all MOVE participants
% mat files are created within each participant folder

%Get a list of all files and folders in this folder.
dataFolder = dir('/data/p_02349/MOVE/');

% Get a logical vector that tells which is a directory.
dirFlags = [dataFolder.isdir];
% Extract only those that are directories.
subjectsFolders = dataFolder(dirFlags);
% Print folder names to command window.
for s = 1 : length(subjectsFolders)
    fprintf('Sub folder #%d = %s\n', s, subjectsFolders(s).name);
end
% Remove other fields and ./.. from folder
subjectsFolders = rmfield(subjectsFolders, {'date', 'bytes', 'isdir', 'datenum'});
subjectsFolders(1:2) = [];


% for loop for each participant with length(subjectsFolders)
for p = 1 : length(subjectsFolders)

    % Extract File names
    fileNames = subjectsFolders(p).name;

    % Change directory
    cd(['/data/p_02349/MOVE/',fileNames,'/KINARM']);

    % check if data_kine.mat already exists in current folder
    % if it does not exist, create it
    if not(isfile('data_kine.mat'))

        % Import Subject data to get ID
        subjFile = importdata('pat.dat');
        subjID = subjFile{11,1}((4:end));
        subjID = str2double(subjID);

        % load data
        data = zip_load();
        % data_mat = data.c3d;

        % if all data (incl. test trials & resting state) is in the directory:
        % only save task 1, 2 & 3 in the matfile
        % order changed after 8 participants
        % participant 9 did practice 2 times
        % participant 56 has task 1 in 2 files + 1 wrongly started task
        % (not resting and practice)
        % participant # 136 has so many error trials that the file got too
        % big
        % participant 31 and 42 have task 3 in 2 files
        if subjID == 1 || subjID == 2 || subjID == 3 || subjID == 8 || ...
           subjID == 11 || subjID == 14 || subjID == 7
            data_task = data([3, 5, 6], :);
        elseif subjID == 9
            data_task = data([4, 6, 7], :);
        elseif subjID == 56
            data_task = data([4, 5, 7, 9], :);
        elseif subjID == 31 || subjID == 42
            data_task = data([3, 5, 7, 8], :);
        else
            data_task = data([3, 5, 7], :);
        end

        % kinematics of data
        % subject 56 has 4 files because task 1 was
        % saved in 2 files
        % partipant 31 and 42 have task 3 in 2 files
        if subjID == 56 || subjID == 31 || subjID == 42
            data_mat_1 = data_task(1).c3d;
            data_mat_2 = data_task(2).c3d;
            data_mat_3 = data_task(3).c3d;
            data_mat_4 = data_task(4).c3d;
            data_mat = [data_mat_1, data_mat_2, data_mat_3, data_mat_4];
        else
            data_mat_1 = data_task(1).c3d;
            data_mat_2 = data_task(2).c3d;
            data_mat_3 = data_task(3).c3d;
            data_mat = [data_mat_1, data_mat_2, data_mat_3];
        end

        % some data of participant 48 was wrongly labeled as participant 33
        % -> changed manually

        % Add estimates of KINARM friction. This function should be called
        % PRIOR to filtering data. When present, these friction estimates will be
        % used by KINARM_add_torques
        %data_fric = KINARM_add_friction(data_mat);

        % Filter all floating-point data. These data include all kinematic data,
        % as well as analog channels. Integer data will not be filtered.
        %data_filter = filter_double_pass(data_fric, 'enhanced', 'fc', 10);
        data_filter = filter_double_pass(data_mat, 'enhanced', 'fc', 10);

        % add hand kinematics: hand velocities, accelerations and commanded
        % forces (in global coordinates) to the data structure. These values
        % are calculated from the joint-based versions that are automatically
        % saved as part of the data
        data_kine = KINARM_add_hand_kinematics(data_filter);

        % Calculate (intramuscular) and applied torques
        % data_final = KINARM_add_torques(data_kine);

        if subjID == 136
            %  (file is too big)
            half_trials = round(length(data_kine)/2);
            data_kine1 = data_kine(:,1:half_trials);
            data_kine2 = data_kine(:,(half_trials+1):(length(data_kine)));
            save("data_kine1.mat", "data_kine1")
            save("data_kine2.mat", "data_kine2")

        else
            % save data_mat as .mat file:
            save("data_kine.mat", "data_kine")
        end
    end

end

