%<FUNCTIONNAME> <Function description.>
%
%  [<outputs>] = <FunctionName>(<inputs>) is for <description>.
%
%  INPUT
%    -<input1>:     <input1 description>
%    -<input2>:     <input2 description>
%
%  OUTPUT
%    -<output1>:    <output2 description>
%
%  Author: Stanislaw Antol (santol@vt.edu)                 Date: 2014-08-18

%% Extract Pose as (x,y) pairs
function [options, partXY] = LoadRelationData(options, inputFile)

inputPose = dlmread(inputFile);
options.numImgs = size(inputPose, 1);
partXY = cell(options.numImgs, options.numPeople);
partXYStruct = struct('x', 'y');

for i = 1:options.numImgs
    
    for j = 1:options.numPeople
        startIdx = (j-1)*2*options.numParts + 1;
        endIdx   =   (j)*2*options.numParts;
        
        partXYStruct.x = inputPose(i, startIdx:2:endIdx)';
        partXYStruct.y = inputPose(i, startIdx+1:2:endIdx+1)';
        
        partXY{i, j} = partXYStruct;
    end
end
end