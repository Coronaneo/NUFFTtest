function [fhat,ffun] = DeCom_NUFFT1D_I(f,x,k,tol)
% This code implements the 1D type I forward NUFFT.
% fhat_j = \sum_{s} f_s exp(-2*pi*i*x_s*k_j)
%
% Input:
% f - a signal of size N by 1, sampled on x
% x - N by 1 samples in time (nonuniform)
% k - N by 1 samples in frequency, k = -N/2:N/2-1 or k = 0:N-1
% tol - accuracy parameter of the NUFFT
% A NUFFT similar to the one proposed in
% "A NONUNIFORM FAST FOURIER TRANSFORM BASED ON LOW RANK APPROXIMATION",
% DIEGO RUIZ?ANTOLIN AND ALEX TOWNSEND, preprint.
%
% Output:
% fhat - the NUFFT of f
% ffun - the function handle for operating the NUFFT, i.e. fhat = ffun(f)
%
% Implemented by Haizhao Yang, 2017.

if nargin < 4, tol = 1e-14; end;
if nargin < 3, k = 0:numel(x)-1; end;
if size(f,1) < 2, f = f(:); end;
if size(x,1) < 2, x = x(:); end;
if size(k,1) < 2, k = k(:); end;

N = numel(x);
if mod(N,2) < 0.5
    if norm(k-(-N/2:(N/2-1))') < eps
        sft = 1;
    else if norm(k-(0:(N-1))') < eps
            sft = 0;
        else
            error('The frequency grid is not correct!');
        end
    end
else
    if norm(k-(-(N-1)/2:(N-1)/2)') < eps
        sft = 1;
    else if norm(k-(0:N-1)') < eps
            sft = 0;
        else
            error('The frequency grid is not correct!');
        end
    end
end
[L,R,idx] = DeCom_NUFFT1D_I_Fac(x,k,tol);
[N,r] = size(L);
Id = sparse(idx,1:N,ones(1,N),N,N);
if sft
    ffun = @(f) sum(L.*fftshift(fft(Id*(R.*repmat(f,[1,r])), [], 1),1),2);
else
    ffun = @(f) sum(L.*fft(Id*(R.*repmat(f,[1,r])), [], 1),2);
end
fhat = ffun(f);
end

function [L,R,t] = DeCom_NUFFT1D_I_Fac(x,k,tol)
% This code implements the low-rank factorization for the 1D type II NUFFT
% in O(N) operations, where N is the length of the signal.
%
% Implemented by Haizhao Yang, 2017.
%
% Input:
% x - samples in time
% k - samples in frequency, k = -N/2:N/2-1 or k = 0:N-1
% tol - accuracy parameter of the NUFFT
%
% Output:
% L - left factor of size N by K
% R - right factor of size N by K
% t - indices for the rounding of x to nearby uniform grid

N = numel(x);
s= round(x*N);
t = mod(s,N)+1;
% gamma - in [0,0.5], quantifying how non-uniform x is
gamma = norm(N*x-s, inf);
% use DIEGO RUIZ?ANTOLIN AND ALEX TOWNSEND's method to estimate the rank
% this rank is good for Chebyshev low-rank approximation but is not tight
% for randomized low-rank factorization
xi = log(log(10/tol)/gamma/7);
lw = xi - log(xi) + log(xi)/xi + .5*log(xi)^2/xi^2 - log(xi)/xi^2;
K = min(16,ceil(5*gamma*exp(lw)));
if gamma < 1e-16 % safe guard
    % DIEGO RUIZ?ANTOLIN AND ALEX TOWNSEND's method is not stable for small gamma
    L = ones(N,1); R = ones(N,1);
else
    % randomized low-rank factorization
    fun = @(k,x) exp(-2*pi*i*k*(x-(mod(round(x*N),N)/N))');
    [U,S,V] = lowrank(k(:),x(:),fun,tol,K,K);
    L = U*S;
    R = conj(V);
    %err = norm(B-L*transpose(R))/norm(B);
    %fprintf('rank of B is %d, approximation error is %.3e\n',K,err);
end
end

function [U,S,V,Ridx,Cidx,rs,cs] = lowrank(x,p,fun,tol,tR,mR)

Nx = size(x,1);
Np = size(p,1);

if(Nx==0 || Np==0)
    U = zeros(Nx,0);
    S = zeros(0,0);
    V = zeros(Np,0);
    Ridx = [];
    Cidx = [];
    rs = [];
    cs = [];
    return;
end

if(tR<Np && tR<Nx)
    %get columns
    rs = randsample(Nx,tR);
    M2 = fun(x(rs,:),p);
    [~,R2,E2] = qr(M2,0);
    Cidx = E2(find(abs(diag(R2))>tol*abs(R2(1)))<=tR);
    
    %get rows
    cs = randsample(Np,tR);
    cs = unique([cs' Cidx]);
    M1 = fun(x,p(cs,:));
    [~,R1,E1] = qr(M1',0);
    Ridx = E1(find(abs(diag(R1))>tol*abs(R1(1)))<=tR);
    
    %get columns again
    rs = randsample(Nx,tR);
    rs = unique([rs' Ridx]);
    M2 = fun(x(rs,:),p);
    [~,R2,E2] = qr(M2,0);
    Cidx = E2(find(abs(diag(R2))>tol*abs(R2(1)))<=tR);
    
    %get rows again
    cs = randsample(Np,tR);
    cs = unique([cs' Cidx]);
    M1 = fun(x,p(cs,:));
    [~,R1,E1] = qr(M1',0);
    Ridx = E1(find(abs(diag(R1))>tol*abs(R1(1)))<=tR);
else
    Ridx = 1:Nx;
    Cidx = 1:Np;
end

%get rows
MR = fun(x(Ridx,:),p);

%get columns
MC = fun(x,p(Cidx,:));

%get middle matrix
[QC,~,~] = qr(MC,0);
[QR,~,~] = qr(MR',0);

if(tR<Np && tR<Nx)
    cs = randsample(Np,tR);
    cs = unique([cs' Cidx]);
    rs = randsample(Nx,tR);
    rs = unique([rs' Ridx]);
else
    cs = 1:Np;
    rs = 1:Nx;
end

M1 = QC(rs,:);
M2 = QR(cs,:);
M3 = fun(x(rs,:),p(cs,:));
MD = pinv(M1) * (M3* pinv(M2'));
[U,S,V] = svdtrunc(MD,mR,tol);
U = QC*U;
V = QR*V;

end

function [U,S,V] = svdtrunc(A,r,tol)

[U,S,V] = svd(A,'econ');
idx = find(find(diag(S)>tol*S(1,1))<=r);
U = U(:,idx);
S = S(idx,idx);
V = V(:,idx);

end

