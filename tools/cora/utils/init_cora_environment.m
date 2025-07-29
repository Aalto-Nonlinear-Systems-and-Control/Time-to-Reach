function init_cora_environment(version_name)
    % INIT_CORA_ENVIRONMENT - Adds a specific CORA version to the path.
    cora_root = fileparts(fileparts(mfilename('fullpath'))); % This is in tools/cora/utils, so go up to tools/cora/
    version_path = fullfile(cora_root, 'versions', version_name);

    if ~exist(version_path, 'dir')
        error('CORA version "%s" not found at path: %s', version_name, version_path);
    end

    % Add the specified version and its subdirectories to the path
    addpath(genpath(version_path));
    fprintf('Successfully added CORA version "%s" to path.\n', version_name);
end
