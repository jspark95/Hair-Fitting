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
inputImage = im2double(imread('03. Input/01. Images/woman3.png'));
% inputImage = im2double(imread('beyonce.jpg'));
width = size(inputImage, 2);
height = size(inputImage, 1);
% mask
maskHair = readNPY('03. Input/02. Masks/woman3.npy');
% maskHair = im2double(imread('beyon_mask.png'));
mask = int8(maskHair == 2);
mask = imresize(mask, [height width]);
mask = double(mask);
% landmark
% load('beyon_lm.mat')
[~, ~, face2Dlm] = mtcnn.detectFaces(inputImage);
% figure, imshow(inputImage)
% hold on, scatter(face2Dlm(1, :, 1), face2Dlm(1, :, 2), 80, 'red', 'filled');

% gender (0 = man, 1 = woman)
genderFlag = 1;

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

tmplm(1,1) = face2Dlm(1,5,1);
tmplm(1,2) = face2Dlm(1,5,2);
tmplm(2,1) = face2Dlm(1,4,1);
tmplm(2,2) = face2Dlm(1,4,2);
tmplm(3,1) = face2Dlm(1,3,1);
tmplm(3,2) = face2Dlm(1,3,2);
tmplm(4,1) = face2Dlm(1,2,1);
tmplm(4,2) = face2Dlm(1,2,2);
tmplm(5,1) = face2Dlm(1,1,1);
tmplm(5,2) = face2Dlm(1,1,2);

face2Dlm = tmplm';
clear tmplm
% face2Dlm = face2Dlm';
face3Dlm = face_v(faceLmIdx, :);
[TR, Tt, Ts] = EstimateSOPwithRefinement(face2Dlm, face3Dlm');

optIdx = 1;
IoUscore = 0;
background = zeros(height, width, 3);
faceColor = repmat([0 0 1], length(face_v), 1);

% Data retrieval
for i = numOfData:numOfData
    disp(['retrieval ', num2str(i)]);
    hair_v = hairObj_v{i};
    hair_f = hairObj_f{i};
    hairNum = length(hair_v);
    hairColor = repmat([1 0 0], hairNum, 1);
    wholeColor = [hairColor; faceColor];
    wPoint = [hair_v; face_v];
    whole_f = [hair_f; face_f + length(hair_v)];
    
    rot_v = TR * wPoint';
    src2D = ((rot_v(1:2, :) + Tt') * Ts)';
    src2D(src2D(:, 1) < 1, 1) = 1;
    src2D(src2D(:, 1) > width, 1) = width;
    src2D(src2D(:, 2) < 1, 2) = 1;
    src2D(src2D(:, 2) > height, 2) = height;
    whole_im = visualize_result( rot_v', src2D, whole_f, width, height, background, wholeColor );
    srcMask = whole_im(:,:,1);
    srcMask(srcMask ~= 0) = 1;
    overlap = double(srcMask & mask);
    range = double(srcMask | mask);
    tmpScore = (sum(sum(overlap)) / sum(sum(range)));
    disp(['score ', num2str(tmpScore)]);
    if IoUscore < tmpScore
        IoUscore = tmpScore;
        optIdx = i;
    end
end

% transformSet optimization
hair_v = hairObj_v{optIdx};
hair_f = hairObj_f{optIdx};
hairNum = length(hair_v);

wPoint = [hair_v; face_v];
whole_f = [hair_f; face_f + length(hair_v)];

TR = vrrotmat2vec(TR);
transformSet = [Ts, TR, Tt];

% [new_transformSet, ~, ~, ~, ~] = lsqnonlin(@(transformSet)EfuncOptimRigidTransform(wPoint, whole_f, hairNum, mask, transformSet, face3Dlm, face2Dlm, Wland, GMPenalty), transformSet, [], [], opts);

% scale = new_transformSet(1);
% rotation = vrrotvec2mat(new_transformSet(2:5));
% translate = new_transformSet(6:7);

scale = transformSet(1);
rotation = vrrotvec2mat(transformSet(2:5));
translate = transformSet(6:7);

rot_v = rotation * wPoint';
src2D = ((rot_v(1:2, :) + translate') * scale)';
src2D(src2D(:, 1) < 1, 1) = 1;
src2D(src2D(:, 1) > width, 1) = width;
src2D(src2D(:, 2) < 1, 2) = 1;
src2D(src2D(:, 2) > height, 2) = height;

result = visualize_result( rot_v', src2D, whole_f, width, height, inputImage, 0 );
imwrite(mask, 'woman4_mask.png');
imwrite(result, 'woman4_result.png');
figure, imshow(result);

function F = EfuncOptimRigidTransform(wPoint, face, hairNum, tarMask, transformSet, lm3, lm2, Wland, GMPenalty)
    width = size(tarMask, 2);
    height = size(tarMask, 1);
    background = zeros(height, width, 3);
    
    hairColor = repmat([1 0 0], hairNum, 1);
    faceNum = length(wPoint) - hairNum;
    faceColor = repmat([0 0 1], faceNum, 1);
    
    wholeColor = [hairColor; faceColor];
    
    % normalize position    
    Ts = transformSet(1);
    TR = vrrotvec2mat(transformSet(2:5));
    Tt = transformSet(6:7);
    
    rot_v = TR * wPoint';
    src2D = ((rot_v(1:2, :) + Tt') * Ts)';
    src2D(src2D(:, 1) < 1, 1) = 1;
    src2D(src2D(:, 1) > width, 1) = width;
    src2D(src2D(:, 2) < 1, 2) = 1;
    src2D(src2D(:, 2) > height, 2) = height;
    
    whole_im = visualize_result( rot_v', src2D, face, width, height, background, wholeColor );
    srcMask = whole_im(:,:,1);
    srcMask(srcMask ~= 0) = 1;
    overlap = double(srcMask & tarMask);
    range = double(srcMask | tarMask);
    E_data = 1 - (sum(sum(overlap)) / sum(sum(range)));
    
    % landmark term
    rotlm3 = TR * lm3;
    projlm3 = ((rotlm3(1:2, :) + Tt') * Ts);
    E_land = projlm3 - lm2;
    E_land = (E_land ./ sqrt(sum(E_land.^2, 2) + (GMPenalty^2))) ./ sqrt(length(E_land));
    
    F = [E_data; sqrt(Wland) .* E_land(:)];
end