function LRNUFFT_startup()
% LRNUFFT_STARTUP  Startup file for LRNUFFT
%   LRNUFFT_STARTUP() adds paths of the LRNUFFT to Matlab.

%   Copyright (c) 2017 Yingzhou Li, Stanford University

file_path = mfilename('fullpath');
tmp = strfind(file_path,'LRNUFFT_startup');
file_path = file_path(1:(tmp(end)-1));

addpath(genpath([file_path 'src']))

end
