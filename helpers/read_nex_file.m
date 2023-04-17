% Function to read a nex file using the NeuroExplorer MATLAB SDK
function [ nexFile ] = read_nex_file(directory, filename)

    % Check if the given file has a .nex or .nex5 extension
    [~,~,ext] = fileparts(filename);
    if ~(strcmp(ext, '.nex') || strcmp(ext, '.nex5'))
        error('The file must have a .nex or .nex5 extension.');
    end

    % Combine the directory and filename
    file_path = fullfile(directory, filename);

    % Read the file using the appropriate function
    if strcmp(ext, '.nex')
        [nexFile] = readNexFile(file_path,'nocont','nowaves');
    elseif strcmp(ext, '.nex5')
        [nexFile] = readNex5File(file_path);
    end
end
