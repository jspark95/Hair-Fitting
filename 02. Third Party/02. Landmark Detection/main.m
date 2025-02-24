im = imread("side.png");
[bboxes, scores, landmarks] = mtcnn.detectFaces(im);
fprintf("Found %d faces.\n", numel(scores));

displayIm = insertObjectAnnotation(im, "rectangle", bboxes, scores, "LineWidth", 2);
figure
imshow(displayIm)
hold on
for iFace = 1:numel(scores)
    scatter(landmarks(iFace, :, 1), landmarks(iFace, :, 2), 'filled');
end