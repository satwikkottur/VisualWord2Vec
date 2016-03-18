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

function out = ComputeFeatures(partXY)
numOrients = 12;
noiseSigma = 0;
loadRelations         = 0;
computeBasic          = 0;
computeContact        = 1;
computeGlobal         = 1;
computeOrient         = 1;

addGaussianNoise      = 0;

if ( nargin == 7 )
    if ( length(noiseSigma) > 0 )
        addGaussianNoise      = 0;
    end
end
options.numImgs = 1;
options.isReal = 0;
options.numGaussians = 3;
options.numParts = 14;
options.numPeople = 1;
options.numJointPairs = 8;
options.numOrient = numOrients;

options.ihead = 0+1;
options.ishoulderL = 2+1;
options.ishoulderR = 5+1;
options.ihipL = 11+1;
options.ihipR = 8+1;

jointPairs0(0+1) = 2+1;
jointPairs0(1+1) = 3+1;
jointPairs0(2+1) = 5+1;
jointPairs0(3+1) = 6+1;
jointPairs0(4+1) = 8+1;
jointPairs0(5+1) = 9+1;
jointPairs0(6+1) = 11+1;
jointPairs0(7+1) = 12+1;
options.numJointPairs0 = length(jointPairs0);
options.jointPairs0 = jointPairs0;

jointPairs1(0+1) =  3+1; % L Elbow
jointPairs1(1+1) =  4+1; % L Hand
jointPairs1(2+1) =  6+1; % R Elbow
jointPairs1(3+1) =  7+1; % R Hand
jointPairs1(4+1) =  9+1; % R Knee
jointPairs1(5+1) = 10+1; % R Foot
jointPairs1(6+1) = 12+1; % L Knee
jointPairs1(7+1) = 13+1; % L Foot
options.numJointPairs1 = length(jointPairs1);
options.jointPairs1 = jointPairs1;

contactJointPairs0(0+1)  =  0+1; % Head
contactJointPairs0(1+1)  =  2+1; % L Shoulder
contactJointPairs0(2+1)  =  3+1; % L Elbow
contactJointPairs0(3+1)  =  4+1; % L Hand
contactJointPairs0(4+1)  =  5+1; % R Shoulder
contactJointPairs0(5+1)  =  6+1; % R Elbow
contactJointPairs0(6+1)  =  7+1; % R Hand
contactJointPairs0(7+1)  =  8+1; % R Hip
contactJointPairs0(8+1)  =  9+1; % R Knee
contactJointPairs0(9+1)  = 10+1; % R Foot
contactJointPairs0(10+1) = 11+1; % L Hip
contactJointPairs0(11+1) = 12+1; % L Knee
contactJointPairs0(12+1) = 13+1; % L Foot
options.numContactJointPairs0 = length(contactJointPairs0);
options.contactJointPairs0 = contactJointPairs0;

jointPairsParent0(0+1) = 2+1;
jointPairsParent0(1+1) = 2+1;
jointPairsParent0(2+1) = 5+1;
jointPairsParent0(3+1) = 5+1;
jointPairsParent0(4+1) = 5+1;
jointPairsParent0(5+1) = 8+1;
jointPairsParent0(6+1) = 2+1;
jointPairsParent0(7+1) = 11+1;
options.jointPairsParent0 = jointPairsParent0;

jointPairsParent1(0+1) = 11+1;
jointPairsParent1(1+1) = 3+1;
jointPairsParent1(2+1) = 8+1;
jointPairsParent1(3+1) = 6+1;
jointPairsParent1(4+1) = 8+1;
jointPairsParent1(5+1) = 9+1;
jointPairsParent1(6+1) = 11+1;
jointPairsParent1(7+1) = 12+1;
options.jointPairsParent1 = jointPairsParent1;

if ( loadRelations ~= 0 )
    [options, partXY] = LoadRelationData(options, inputDataFile);
end

if ( computeBasic ~= 0 )
    basicfeatures = ComputeBasicFeatures(options, partXY);
    %name = sprintf('%s_basicFeatures', inputDataFile(1:end-4));
    %SaveFeatures(name, basicfeatures);
end

for numOrient = numOrients
    options.numOrient = numOrient;
    if ( computeOrient ~= 0 )
        orientFeatures = ComputeOrientationFeatures(options, partXY);
        %name = sprintf('%s_orient%02dFeatures', inputDataFile(1:end-4), options.numOrient);
        %SaveFeatures(name, orientFeatures);
    end
end

if ( computeGlobal ~= 0 )
    globalFeatures = ComputeGlobalPositionFeatures(options, 0, partXY, 0, 0);
    %name = sprintf('%s_globalPositionFeatures', inputDataFile(1:end-4));
    %SaveFeatures(name, globalFeatures);
end

if ( computeContact ~= 0 )
    contactFeatures = ComputeContactFeatures(options, partXY);
    %name = sprintf('%s_contactFeatures', inputDataFile(1:end-4));
    %SaveFeatures(name, contactFeatures);
end

out = cat(2,orientFeatures,globalFeatures,contactFeatures);

if ( addGaussianNoise ~= 0 )
    % PARSE
%     noiseSigma = [0.00, 0.10, 0.25, 0.5, 0.75, 1.0, 2.0];
    randSeedVals = 1;
    randSeedValsNum = 10;

%    % Ours
%     noiseSigma = [0.0, 2.0];
%     randSeedVals = 1;
%     randSeedValsNum = 1;
    
    for sigmaIdx = 1:length(noiseSigma)
        sigma = noiseSigma(sigmaIdx);
        
        for randIdx = 1:randSeedValsNum
            options.seedVal = randSeedVals + randIdx;
            
            partXYNoise = AddGaussianNoise(options, partXY, sigma);
            
            poseDataNoise = zeros(size(partXYNoise, 1), 2*options.numParts*options.numPeople);
            
%              poseData1 = [reshape(avgPoses(1).pose', 2*options.numParts, 1);
%                      reshape(avgPoses(2).pose', 2*options.numParts, 1)];
            
            for i = 1:size(partXYNoise, 1)
                poseData = [];
                for j = 1:options.numPeople
%                     for k = 1:options.numParts
%                         poseData = [poseData, partXYNoise{i, j}.x(k), partXYNoise{i, j}.y(k)];
%                     end
                    poseData = [poseData, reshape([partXYNoise{i, j}.x, partXYNoise{i, j}.y]', 1, 14*2)];
                end
                
                poseDataNoise(i, :) = poseData;
            end
            
            name = sprintf('%s_sigma_%05.3f_seed_%05d.txt', inputDataFile(1:end-4), sigma, options.seedVal);
            dlmwrite(name, poseDataNoise, ';');        

        end
    end
end


end