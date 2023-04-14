
% Initialize the directory and file variables
file = 'SFN16_2019-03-25_SF.nex';
dir = 'C:\repos\neuroexplore\data';

% Create a valid filename by combining directory and file
fullFileName = fullfile(dir, file);

% Check if 'nex' or 'nex5' is present in the fullFileName
if contains(fullFileName, 'nex5')
    version = 2;
elseif contains(fullFileName, 'nex')
    version = 1;
else
    version = 0;
end

if version==1 
    nex = readNexFile(fullFileName,'nocont','nowaves');
elseif version==2 
    nex = readNex5File(fullFileName);
else
    error('Error: Invalid version!'); 
end
nex
if isfield(nex,'waves')
    nex = rmfield(nex,'waves');
end

if isfield(nex,'contvars')
    nex = rmfield(nex,'contvars');
end

% Turn "events" struct into table and rid of blanks
nex.events = struct2table(cat(1,nex.events{:}),'AsArray',true);
nex.events.name = deblank(nex.events.name); 

nex.events(:,2) = []; % Get rid of unused "var version" table column

% ensure that all "neuron" variables have the same fields
nex.neurons = cellfun(@(x) struct('name',x.name,'timestamps',x.timestamps), nex.neurons, 'UniformOutput', false);
nex.neurons = struct2table(cat(1,nex.neurons{:}),'AsArray', true);
nneur = height(nex.neurons);


% end