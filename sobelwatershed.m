function [ L ] = sobelwatershed( imname )
%SOBELWATERSHED 

rgb = imread(imname);
%rgb = imresize(rgb,[512,512]);
%imshow(rgb);
if ndims(rgb)==3
    I = rgb2gray(rgb);
else
    I = rgb;
end
verbose = 0;
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
if verbose==1
    figure
    imshow(gradmag,[]), title('Gradient magnitude (gradmag)')
end

L = watershed(gradmag);
Lrgb = label2rgb(L);
if verbose==1
    figure, imshow(Lrgb), title('Watershed transform of gradient magnitude (Lrgb)')
end

se = strel('disk',1);
Io = imopen(I, se);
if verbose==1
    figure
    imshow(Io), title('Opening (Io)')
end

Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
if verbose==1
    figure
    imshow(Iobr), title('Opening-by-reconstruction (Iobr)')
end

Ioc = imclose(Io, se);
if verbose==1
    figure
    imshow(Ioc), title('Opening-closing (Ioc)')
end

Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
if verbose==1
    figure
    imshow(Iobrcbr), title('Opening-closing by reconstruction (Iobrcbr)')
end

fgm = imregionalmax(Iobrcbr);
if verbose==1
    figure
    imshow(fgm), title('Regional maxima of opening-closing by reconstruction (fgm)')
end

I2 = I;
I2(fgm) = 255;
if verbose==1
    figure
    imshow(I2), title('Regional maxima superimposed on original image (I2)')
end

se2 = strel(ones(5,5));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);

fgm4 = bwareaopen(fgm3, 5);
I3 = I;
I3(fgm4) = 255;
if verbose==1
    figure
    imshow(I3)
    title('Modified regional maxima superimposed on original image (fgm4)')
end

bw = imbinarize(Iobrcbr);
if verbose==1
    figure
    imshow(bw), title('Thresholded opening-closing by reconstruction (bw)')
end

D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
if verbose==1
    figure
    imshow(bgm), title('Watershed ridge lines (bgm)')
end

gradmag2 = imimposemin(gradmag, bgm | fgm4);
L = watershed(gradmag2);
savename = strcat(imname(1:end-3),'mat');
label = L;
save(savename,'label');
% minlab = min(min(L));
% maxlab = max(max(L));
% for i = minlab:maxlab
%     mask = zeros(size(L));
%     [row,col] = find(L==i);
%     for j = 1:size(row,1)
%         mask(row(j),col(j)) = 1;
%     end
%     figure,imshow(mask);
% end
I4 = I;
I4(imdilate(L == 0, ones(3, 3)) | bgm | fgm4) = 255;
if verbose==1
    figure
    imshow(I4)
    title('Markers and object boundaries superimposed on original image (I4)')
end

Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
if verbose==1
    figure
    imshow(Lrgb)
    title('Colored watershed label matrix (Lrgb)')
end

if verbose==1
    figure
    imshow(I)
    hold on
    himage = imshow(Lrgb);
    himage.AlphaData = 0.3;
    title('Lrgb superimposed transparently on original image')
end

end

