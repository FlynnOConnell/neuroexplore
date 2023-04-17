%   [S,t,f,R,Serr]=mtspecgrampt(data,movingwin,params,fscorr)
%
%   [ data ]: matrix in the form of times * trials/channels
%
%   [ winsize winstep ]: probably 200 to 500 ms will be good and step aka
%       the overlap fraction probably no overlap or 50% overlap
%
%   [ paraps ] struct with fields
%   [ params.taper ]: [NW K] = [3 5] default  better resolution
%       with [NW K]=[2 3] or [NW K] = [1 1]). 
%   [ params.Fs ]: Sample frequency

params = struct();

params.tapers = [2 3];
params.Fs = 1000;
params.fpass = [0 params.Fs/2];

window_length = 0.2; % 200 ms window
window_overlap = 0; % 0% overlap
step_size = window_length * (1 - window_overlap);
movingwin = [window_length, step_size];

for fieldname = fieldnames(e_BR)'
    e_BR.(fieldname{1}) = e_BR.(fieldname{1}).';
end

data = struct();
data.times = {e_BR.A, e_BR.B}

[S,t,f,Serr] = mtspecgrampt_optimized( e_BR, movingwin, params );