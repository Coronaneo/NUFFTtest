function nufftfun = nufftI(x,iflag,ns,rt,tol)
% NUFFTI: Computation of nonuniform FFT in R^1 - Type I.
%   gfun = NUFFTI(x,iflag,ns,rt,tol) provides the fast algorithm for nufft
%   of type I. x is the location of sources on interval [0,ns-1], iflag
%   determines the sign of FFT, and ns is the number of Fourier modes
%   computed. The function returns a function that fast evaluate nufft.
%
%   NUFFT of type I is defined as follows.
%              ns
%     g(k) = w sum c(j) exp(+/-i 2pi * k x(j) / ns)
%              j=1
%   for 1 <= k <= ns and 1 <= x(j) <= ns.
%   If (iflag .ge.0) the + sign is used in the exponential and a = 1/ns.
%   If (iflag .lt.0) the - sign is used in the exponential and a = 1.
%
%   See also FFT, NUFFTII, NUFFTIII.


if nargin < 2
    iflag = -1;
else
    iflag = sign(iflag);
end

if nargin < 3
    ns = ceil(max(x))+1;
end

if nargin < 4
    rt = 15;
end

if nargin < 5
    tol = 1e-6;
end

k = (0:ns-1)';

fftconst = iflag*1i/ns*2*pi;

ratiofun = @(k,x)exp(fftconst*k*(x-round(x))');
[U,V] = lowrank(k,x,ratiofun,tol,rt,rt);

r = size(V,2);

xsub = mod(round(x),ns)+1;
spPerm = sparse(xsub,1:ns,ones(1,ns),ns,ns);

nufftfun = @(c)nufftIfun(c);

    function fftc = nufftIfun(c)
        [n,ncol] = size(c);

        c = repmat(conj(V),1,ncol).*reshape(repmat(c,r,1),n,r*ncol);
        c = spPerm*c;
        if iflag < 0
            fftc = fft(c);
        else
            fftc = ifft(c);
        end
        fftc = squeeze(sum(reshape(repmat(U,1,ncol).*fftc,n,r,ncol),2));
    end

end