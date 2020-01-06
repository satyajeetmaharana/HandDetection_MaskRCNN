videos = getMetaBy();
for i = 1:48
for j = 1:100
img = imread(getFramePath(videos(i),j));
hand_mask = getSegmentationMask(videos(i),j,'all');
bounding_boxes = getBoundingBoxes(videos(i),j);
str = sprintf('DATA_IMAGES/Image%d_%d.jpg',i,j);
imwrite(img,str);
str2 = sprintf('DATA_MASKS/Mask%d_%d.jpg',i,j);
imwrite(hand_mask,str2);
str3 = sprintf('DATA_BOXES/Box%d_%d.csv',i,j);
csvwrite(str3,bounding_boxes);
end
end