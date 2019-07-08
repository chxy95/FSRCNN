% =========================================================================
% Test code for Fast Super-Resolution Convolutional Neural Networks (FSRCNN)
%
% Reference
%   Chao Dong, Chen Change Loy, Xiaoou Tang. Accelerating the Super-Resolution Convolutional Neural Networks, 
%   in Proceedings of European Conference on Computer Vision (ECCV), 2016
%
% Chao Dong
% SenseTime Group
% For any question, send email to chaodong@sensetime.com
% =========================================================================
close all;
clear all;

%% set parameters
testfolder = 'test\Set5\';
up_scale = 3;
%model = 'model\FSRCNN-s\x3.mat';
model = 'mine.mat';

filepaths = dir(fullfile(testfolder,'*.bmp'));
psnr_bic = zeros(length(filepaths),1);
psnr_fsrcnn = zeros(length(filepaths),1);

for i = 1 : length(filepaths)
   
    %% read ground truth image
    [add,imname,type] = fileparts(filepaths(i).name);
    im = imread([testfolder imname type]);

    %% work on illuminance only
    if size(im,3) > 1
        im_ycbcr = rgb2ycbcr(im);
        
        im_cbcr = im_ycbcr(:, :, 2:3);
        im = im_ycbcr(:, :, 1);
    end
    im_gnd = modcrop(im, up_scale);
    im_gnd = single(im_gnd)/255;
    im_l = imresize(im_gnd, 1/up_scale, 'bicubic');

    %% FSRCNN
    im_h = FSRCNN(model, im_l, up_scale);

    %% bicubic interpolation
    im_b = imresize(im_l, up_scale, 'bicubic');

    %% remove border
    if up_scale==3
        im_h = shave_x3(uint8(im_h * 255), [up_scale, up_scale]);
    else 
        im_h = shave(uint8(im_h * 255), [up_scale, up_scale]);
    end
    im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
    im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);
    
    im_cbcr =  modcrop(im_cbcr, up_scale);
    im_cbcr = shave(im_cbcr, [up_scale, up_scale]);
    
    img1 = cat(3, im_b, im_cbcr);
    img1 = ycbcr2rgb(img1);
    %figure, imshow(img1); title('Bicubic Interpolation');

    img2 = cat(3, im_h, im_cbcr);
    img2 = ycbcr2rgb(img2);
    %figure, imshow(img2); title('SRCNN Reconstruction');

    img3 = cat(3, im_gnd, im_cbcr);
    img3 = ycbcr2rgb(img3);
    %figure, imshow(img3); title('gnd');
    %% compute PSNR
    psnr_bic(i) = compute_psnr(im_gnd,im_b);
    psnr_fsrcnn(i) = compute_psnr(im_gnd,im_h);
    
    fprintf('%s PSNR: %f dB\n', imname, psnr_fsrcnn(i));
    %% save results
    %imwrite(img1, ['result/' imname '_bic.bmp']);
    %imwrite(img2, ['result/' imname '_FSRCNN.bmp']);
    %imwrite(img2, ['result/' imname '_DSRCNN.bmp']);
    %imwrite(img3, ['result/' imname '_ori.bmp']);

end

fprintf('Mean PSNR for Bicubic: %f dB\n', mean(psnr_bic));
fprintf('Mean PSNR for FSRCNN: %f dB\n', mean(psnr_fsrcnn));
