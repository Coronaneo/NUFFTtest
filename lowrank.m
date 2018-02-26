function [U,V] = lowrank(x,y,fun,tol,tR,mR)

Nx = size(x,1);
Ny = size(y,1);

if Nx==0 || Ny==0
    U = zeros(Nx,0);
    V = zeros(Ny,0);
    return;
end

if(tR<Ny && tR<Nx)
    %get columns
    rs = randsample(Nx,tR);
    M2 = fun(x(rs,:),y);
    [~,R2,E2] = qr(M2,0);
    Cidx = E2(find(abs(diag(R2))>tol*abs(R2(1)))<=tR);

    %get rows
    cs = randsample(Ny,3);
    cs = unique([cs' Cidx]);
    M1 = fun(x,y(cs,:));
    [~,R1,E1] = qr(M1',0);
    Ridx = E1(find(abs(diag(R1))>tol*abs(R1(1)))<=tR);

    %get columns again
    rs = randsample(Nx,3);
    rs = unique([rs' Ridx]);
    M2 = fun(x(rs,:),y);
    [~,R2,E2] = qr(M2,0);
    Cidx = E2(find(abs(diag(R2))>tol*abs(R2(1)))<=tR);

else
    Ridx = 1:Nx;
    Cidx = 1:Ny;
end

%get rows
MR = fun(x(Ridx,:),y);

%get columns
MC = fun(x,y(Cidx,:));

%get middle matrix
[QC,~,~] = qr(MC,0);
[QR,~,~] = qr(MR',0);

if(tR<Ny && tR<Nx)
    cs = randsample(Ny,tR);
    cs = unique([cs' Cidx]);
    rs = randsample(Nx,tR);
    rs = unique([rs' Ridx]);
else
    cs = 1:Ny;
    rs = 1:Nx;
end

M1 = QC(rs,:);
M2 = QR(cs,:);
M3 = fun(x(rs,:),y(cs,:));
MD = pinv(M1) * (M3* pinv(M2'));
[U,S,V] = svdtrunc(MD,mR,tol);
U = QC*U*sqrt(S);
V = QR*V*sqrt(S);

end