% Author: Ramakrishna Vedantam
% Date : 04-18-2015 3.34 pm

% Code to load the mat files with locations
% Learn a GMM based on those locations with 9 x 5 = 40 mixtures for location and 24 mixtures for relative
% location
% TO-DO: Adjust scale for park scenes if any, presently not incorporated 
clear all; close all; clc;
params;

%% learn a GMM for absolute locations
load('occur_coords_fullvqa.mat');
depths = unique(z);

assert(size(x,1)==1)
assert(size(y,1)==1)
assert(size(z,1)==1)
assert(size(flip,1)==1)

for i = 1:length(depths)
	I = z==depths(i);
	coords = [x(I)', y(I)'];
	GAbsPos(i) = kmeansGMMFit(coords, AbsNComponents)
end

save('../absGMMFull.mat', 'GAbsPos');

%% learn a GMM for relative locations
load('cooccur_coords_fullvqa.mat');
assert(size(x1,1)==1)
assert(size(y1,1)==1)
assert(size(z1,1)==1)
assert(size(x2,1)==1)
assert(size(y2,1)==1)
assert(size(z2,1)==1)
assert(size(flip,1)==1)
% adjust for depth scaling
zfact = ones(1,5);
% assumes only Indoor Scenes - parks have different scaling
zt = 0.95;
for i=2:1:5
    zfact(i) = zfact(i-1)*zt;
end

% convert flip into +1 and -1
flip(flip==1) = -1;
flip(flip==0) = 1;

flip = double(flip);
z1 = double(z1);
zfact = double(zfact);
x1 = double(x1);
x2 = double(x2);
y1 = double(y1);
y2 = double(y2);

del_x = ((x1-x2).*flip)./zfact(z1+1);
del_y = (y1-y2)./zfact(z1+1); 

coords = [del_x', del_y'];

GRelPos = kmeansGMMFit(coords, RelNComponents)

save('../relGMMFull.mat', 'GRelPos');



