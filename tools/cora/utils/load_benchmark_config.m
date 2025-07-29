function config = load_benchmark_config(file_path)
    % Reads a JSON configuration file and returns a MATLAB struct.
    if ~exist(file_path, 'file')
        error('Configuration file not found: %s', file_path);
    end

    fid = fopen(file_path);
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    config = jsondecode(str);
end
