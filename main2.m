% Main Code
% Fitting Hair Model to Image
% Using Similarity Fitting
% Optimization, lsqnonlin
% transformSet = [scale rotation translation]

% Add Path
addpath('02. Third Party/01. npy-matlab')
addpath('02. Third Party/02. Landmark Detection')
addpath('02. Third Party/03. SOP')

% Collect Database
man_listing = dir('01. Database/SIMS/obj/man_obj/*.obj');
woman_listing = dir('01. Database/SIMS/obj/woman_obj/*.obj');

% % Man's Hair Listing
% for i = 1:length(man_listing)
%     tmpObj = readObj([man_listing(i).folder, '/' man_listing(i).name]);
%     manHairObj_v{i} = tmpObj.v; % vertices
%     manHairObj_f{i} = tmpObj.f.v; % faces
% end
% 
% % Woman's Hair Listing
% for i = 1:length(woman_listing)
%     tmpObj = readObj([woman_listing(i).folder, '/' woman_listing(i).name]);
%     womanHairObj_v{i} = tmpObj.v; % vertices
%     womanHairObj_f{i} = tmpObj.f.v; % faces
% end

% input data
% image
inputImage = im2double(imread('03. Input/01. Images/man1.png'));
width = size(inputImage, 2);
height = size(inputImage, 1);
% mask
maskHair = readNPY('03. Input/02. Masks/mask.npy');
mask = int8(maskHair == 2);
mask = imresize(mask, [height width]);
mask = double(mask);
% landmark
[~, ~, face2Dlm] = mtcnn.detectFaces(inputImage);
% gender (0 = man, 1 = woman)
genderFlag = 0;

if (genderFlag)
    load('woman_lm_idx.mat') % face 3D landmark index, faceLmIdx
    load('woman_hair_v.mat') % hairObj_v
    load('woman_hair_f.mat') % hairObj_f
    faceObj = readObj('woman_face.obj');
    numOfData = length(hairObj_v);
else
    load('man_lm_idx.mat') % face 3D landmark index, faceLmIdx
    load('man_hair_v.mat') % hairObj_v
    load('man_hair_f.mat') % hairObj_f
    faceObj = readObj('man_face.obj');
    numOfData = length(hairObj_v);
end

% facial vertex & face
face_v = faceObj.v;
face_f = faceObj.f.v;

% optimizer setting
% opts = optimoptions(@lsqnonlin, 'Display', 'off');
opts = optimoptions(@fmincon, 'Display', 'iter');
Wland = 0.1;
GMPenalty = 0.0001;

% initial transformSet
tmplm = zeros(5,2);
tmplm(:,1) = face2Dlm(1,:,1);
tmplm(:,2) = face2Dlm(1,:,2);
face2Dlm = tmplm';
clear tmplm

face3Dlm = face_v(faceLmIdx, :);
[TR, Tt, Ts] = EstimateSOPwithRefinement(face2Dlm, face3Dlm');
% test
rot_v = TR * face3Dlm';
src2D = ((rot_v(1:2, :) + Tt')* Ts )';