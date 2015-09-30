function [ match_score ] = x( )
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Read the Image %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1=imread(input('Enter Image 1'));
b1=imread(input('Enter Image 2'));
    a=a1(1:284,1:384);
    b=b1(1:284,1:384);
    size_a=size(a);
    size_b=size(b);

    %%%%%%%%%%%%%%%%%%%%%%%%%%% Center of gravity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [c1_a c2_a]=cent_o(a);
    [c1_b c2_b]=cent_o(b);

    %%%%%%%%%%%%%%%%%%%%%%% 2d hanning windowing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for y=-c2_b+1:size_b(2)-c2_b
        for z=-c1_b+1:size_b(1)-c1_b
            wb(z+c1_b,y+c2_b)=((1+cos((pi*z)/162))/2)*((1+cos((pi*y)/162))/2);
            bw(z+c1_b,y+c2_b)=b(z+c1_b,y+c2_b)*wb(z+c1_b,y+c2_b);
        end
    end
    for y=-c2_a+1:size_a(2)-c2_a
        for z=-c1_a+1:size_a(1)-c1_a
            wa(z+c1_a,y+c2_a)=((1+cos((pi*z)/162))/2)*((1+cos((pi*y)/162))/2);
            aw(z+c1_a,y+c2_a)=a(z+c1_a,y+c2_a)*wa(z+c1_a,y+c2_a);
        end
    end


    %%%%%%%%%%%%%%%%%%% scaling and rotation normalization %%%%%%%%%%%%%%%%%%%%
    A=fftshift(fft2(aw));
    B=fftshift(fft2(bw));

    mod_A=sqrt(abs(A));
    mod_B=sqrt(abs(B));


    SizeX=size_a(1);
    SizeY=size_a(2);
    L1 = transformImage(mod_A, SizeX, SizeY, SizeX, SizeY, 'bicubic', size(mod_A) / 2, 'full');
    L2 = transformImage(mod_B, SizeX, SizeY, SizeX, SizeY, 'bicubic', size(mod_B) / 2, 'full');

    THETA_F1 = fft2(L1);
    THETA_F2 = fft2(L2);

    a1 = angle(THETA_F1);
    a2 = angle(THETA_F2);

    THETA_CROSS = exp(1i * (a1 - a2));
    THETA_PHASE = real(ifft2(THETA_CROSS));

    THETA_SORTED = sort(THETA_PHASE(:));  % TODO speed-up, we surely don't need to sort
    SI = length(THETA_SORTED):-1:length(THETA_SORTED);
    [THETA_X, THETA_Y] = find(THETA_PHASE == THETA_SORTED(SI));

    DPP = 360 / size(THETA_PHASE, 2);
    Theta = DPP * (THETA_Y - 1);

    if (size(Theta,1)== 0)
        Theta = 0;
    end

    b_sr=imrotate(b,-Theta,'nearest','crop');

    %%%%%%%%%%%%%%%%%%%%%%% Translation Alignment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    red_rat=100;
    M1=128;
    M2=128;
    K1=floor(M1*red_rat/100);
    K2=floor(M2*red_rat/100);

    a_new=a(142-K1:142+K1,192-K2:192+K2);
    b_new=b_sr(142-K1:142+K1,192-K2:192+K2);
    rfg=blpoc(a_new,b_new);
    k=1:2*K1+1; j=1:2*K2+1; 
    
    rfg_sorted = sort(rfg(:));
    ind=length(rfg_sorted):-1:length(rfg_sorted);
    [delta1_ delta2_]=find(rfg == rfg_sorted(ind));
    delta1=delta1_*((2*M1+1)/(2*K1+1));
    delta2=delta2_*((2*M2+1)/(2*K2+1));
    
    if (delta1 > 128)
        delta1=259-delta1; 
        flag1=1;
    else
        flag1=-1;
    end

    if (delta2 > 128)
        delta2=259-delta2;
        flag2=1;
    else
        flag2=-1;
    end
    b_srt=circshift(b_sr,[flag1*(delta1-1) flag2*(delta2-1)]);
    %%%%%%%%%%%%%%%%%%%% Effective Region Extraction %%%%%%%%%%%%%%%%%%%%%%%%%%
tic
    [c1_sbrt c2_sbrt]=cent_o(b_srt);
    mean_c1=ceil((c1_a+c1_sbrt)/2);
    mean_c2=ceil((c2_a+c2_sbrt)/2);
    m1=90; m2=90;
    f_ddash=a(mean_c1-m1/2:mean_c1+m1/2,mean_c2-m2/2:mean_c2+m2/2);
    g_ddash=b_srt(mean_c1-m1/2:mean_c1+m1/2,mean_c2-m2/2:mean_c2+m2/2);
    rfg_ddash=blpoc(f_ddash,g_ddash);
    h=-m1/2:m1/2-1;
    h1=-m1/2:m1/2-1;
    rfg_sorted=sort(rfg_ddash(:));
    p=4;
    match_score=0;
    for q=1:p
        match_score=match_score+rfg_sorted(length(rfg_sorted)+1-q);
    end
toc
match_score
end

function [r]=blpoc(a,b)
AF=fft2(a);
BF=fft2(b);
[m,n]=size(AF);
Rfg=zeros(m,n);
for k=1:m
    for j=1:n
        temp1=conj(AF(k,j));
        Rfg(k,j)=(BF(k,j)*temp1)/(abs(BF(k,j)*temp1));
    end
end
r=ifft2(Rfg);
end

function [ c1_a, c2_a ]=cent_o( a )
size_a=size(a);
pn2_a=zeros(1,size_a(1));
for i=1:size_a(1)
    for j=1:size_a(2)
        pn2_a(i)=pn2_a(i)+a(i,j)/size_a(2);
    end
end
pn1_a=zeros(1,size_a(2));
for i=1:size_a(2)
    for j=1:size_a(1)
        pn1_a(i)=pn1_a(i)+a(j,i)/size_a(1);
    end
end
mu_pn2_a=mean(pn2_a);
mu_pn1_a=mean(pn1_a);
c1_a=(max(find(pn2_a>=mu_pn2_a))+min(find(pn2_a>=mu_pn2_a)))/2;
c2_a=(max(find(pn1_a>=mu_pn1_a))+min(find(pn1_a>=mu_pn1_a)))/2;

end



