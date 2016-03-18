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

%% Compute Orientation Features
function orientFeatures = ComputeOrientationFeatures(options, partXY)

orientFeatures = zeros(options.numImgs, options.numOrient*options.numPeople*(options.numJointPairs));

% orientFeatures = [];
for i = 1:options.numImgs
    fvec = [];
    
    for j = 1:options.numPeople
        hist = -1*ones(1, options.numOrient);
        midShoulder.x = -1;
        midShoulder.y = -1;
        head.x = -1;
        head.y = -1;
        
        personPose = partXY{i, j};
        
        if (    personPose.x(options.ihead) == -1 || ...
                personPose.y(options.ihead) == -1 || ...
                personPose.x(options.ishoulderL) == -1 || ...
                personPose.y(options.ishoulderL) == -1 || ...
                personPose.x(options.ishoulderR) == -1 || ...
                personPose.y(options.ishoulderR) == -1 )
            
            fvec = [fvec, -1*ones(1, options.numOrient*(options.numJointPairs))];
        else
            midShoulder.x = ( personPose.x(options.ishoulderL) + personPose.x(options.ishoulderR) ) / 2.0;
            midShoulder.y = ( personPose.y(options.ishoulderL) + personPose.y(options.ishoulderR) ) / 2.0;
            
            head.x = personPose.x(options.ihead);
            head.y = personPose.y(options.ihead);
            
            orienBody = single( single(pi) + atan2(head.y - midShoulder.y, head.x - midShoulder.x) );
            
            orienIdx = single( orienBody / single( single(2.0) * pi ) );
            orienIdx = single( orienIdx * single(options.numOrient-single(0.001)) );
            
            orienDel = single( orienIdx - single(floor(orienIdx)) );
            
            hist = zeros(1, options.numOrient);
            
            idx1 = floor(orienIdx) + 1;
            idx2 = mod(idx1, options.numOrient) + 1;
            hist( idx1 ) = single( single(1.0) - orienDel );
            hist( idx2 ) = orienDel;
            
%             fvec = [fvec, hist];
            
            for k = 1:options.numJointPairs
                if (personPose.x(options.jointPairs0(k)) == -1.0 || ...
                        personPose.y(options.jointPairs0(k)) == -1.0 || ...
                        personPose.x(options.jointPairs1(k)) == -1.0 || ...
                        personPose.y(options.jointPairs1(k)) == -1.0 || ...
                        personPose.x(options.jointPairsParent0(k)) == -1.0 || ...
                        personPose.y(options.jointPairsParent0(k)) == -1.0 || ...
                        personPose.x(options.jointPairsParent1(k)) == -1.0 || ...
                        personPose.y(options.jointPairsParent1(k)) == -1.0 ...
                        )
                    
                    fvec = [fvec, -1*ones(1, options.numOrient)];
                else
                    
                    orienParent = pi + atan2( personPose.y(options.jointPairsParent1(k)) - personPose.y(options.jointPairsParent0(k)), ...
                        personPose.x(options.jointPairsParent1(k)) - personPose.x(options.jointPairsParent0(k)) );
                    
                    orien = pi + atan2(personPose.y(options.jointPairs1(k)) - personPose.y(options.jointPairs0(k)), personPose.x(options.jointPairs1(k)) - personPose.x(options.jointPairs0(k)));
                    
                    if (orien < orienParent)
                        orien = orien + 2.0*pi;
                    end
                    
                    orien = orien - orienParent;
                    
                    orienIdx = orien / (2.0 * pi);
                    orienIdx = orienIdx*(options.numOrient - single(0.001));
                    
                    orienDel = orienIdx - floor(orienIdx);
                    
                    hist = zeros(1, options.numOrient);
                    
                    idx1 = floor(orienIdx) + 1;
                    idx2 = mod(idx1, options.numOrient) + 1;
                    hist( idx1 ) = 1.0 - orienDel;
                    hist( idx2 ) = orienDel;
                    
                    fvec = [fvec, hist];
                end
            end
        end
    end
    try
        if ( ~isempty(fvec) )
            orientFeatures(i, :) = fvec;
        end
    catch
        i
        whos
        break
    end
end

end