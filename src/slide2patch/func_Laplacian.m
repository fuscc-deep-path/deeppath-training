%% input an image, output the value for blurring.
function value = func_Laplacian(img)
    gray = double(rgb2gray(img));
    [M,N]=size(gray);
    B=zeros(size(gray));
    for x=2:M-1
        for y=2:N-1
            B(x,y)=gray(x+1,y)+gray(x-1,y)+gray(x,y+1)+gray(x,y-1)-4*gray(x,y);
        end
    end
%     B=im2uint8(B);
    value = var(reshape(B,[],1));
end
