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

%% Compute Contact Features
function contactFeatures = ComputeContactFeatures(options, partXY)


if ( options.numPeople > 1 )
    contactFeatures = zeros(options.numImgs, 2*options.numPeople*options.numJointPairs*options.numContactJointPairs0);
else
    contactFeatures = zeros(options.numImgs, options.numPeople*options.numJointPairs*options.numContactJointPairs0);
end

avgScale = 0.0;
numScale = 0;
for i = 1:options.numImgs
    for j = 1:options.numPeople
        baseBodyIdx = mod( j-1, options.numPeople ) + 1;
        IJBodyIdx   = mod( j  , options.numPeople ) + 1;
        personBase = partXY{i, baseBodyIdx};
        
        mean = [struct('x', 'y'); struct('x', 'y')];
        
        if (personBase.x(options.ihead) == -1.0 || ...
                personBase.y(options.ihead) == -1.0 || ...
                personBase.x(options.ishoulderL) == -1.0 || ...
                personBase.y(options.ishoulderL) == -1.0 || ...
                personBase.x(options.ishoulderR) == -1.0 || ...
                personBase.y(options.ishoulderR) == -1.0 || ...
                personBase.x(options.ihipL) == -1.0 || ...
                personBase.y(options.ihipL) == -1.0 || ...
                personBase.x(options.ihipR) == -1.0 || ...
                personBase.y(options.ihipR) == -1.0 ...
                )
            % do nothing
            
        else
            mean(1+1).x = (personBase.x(options.ishoulderL) + personBase.x(options.ishoulderR) + personBase.x(options.ihipL) + personBase.x(options.ihipR)) / 4.0;
            mean(1+1).y = (personBase.y(options.ishoulderL) + personBase.y(options.ishoulderR) + personBase.y(options.ihipL) + personBase.y(options.ihipR)) / 4.0;
            
            mean(0+1).x = personBase.x(options.ihead);
            mean(0+1).y = personBase.y(options.ihead);
            
            scaleBody = sqrt((mean(1+1).x - mean(0+1).x)*(mean(1+1).x - mean(0+1).x) + (mean(1+1).y - mean(0+1).y)*(mean(1+1).y - mean(0+1).y));
            avgScale = avgScale + scaleBody;
            numScale = numScale+1;
        end
    end
end

avgScale = avgScale/numScale;
%fprintf('Average body scale = %f\n', avgScale);

for i = 1:options.numImgs
    fvec = [];
    for j = 1:options.numPeople
        
        baseBodyIdx = mod( j-1, options.numPeople ) + 1;
        IJBodyIdx   = mod( j  , options.numPeople ) + 1;
        
        personIJ   = partXY{i, IJBodyIdx};
        personBase = partXY{i, baseBodyIdx};
        
        mean = [struct('x', 'y'); struct('x', 'y')];
        scaleBody = avgScale;
        
        if (    personBase.x(options.ihead) == -1.0 || ...
                personBase.y(options.ihead) == -1.0 || ...
                personBase.x(options.ishoulderL) == -1.0 || ...
                personBase.y(options.ishoulderL) == -1.0 || ...
                personBase.x(options.ishoulderR) == -1.0 || ...
                personBase.y(options.ishoulderR) == -1.0 || ...
                personBase.x(options.ihipL) == -1.0 || ...
                personBase.y(options.ihipL) == -1.0 || ...
                personBase.x(options.ihipR) == -1.0 || ...
                personBase.y(options.ihipR) == -1.0 ...
                )
            % do nothing
            
        else
            mean(1+1).x = (personBase.x(options.ishoulderL) + personBase.x(options.ishoulderR) + personBase.x(options.ihipL) + personBase.x(options.ihipR)) / 4.0;
            mean(1+1).y = (personBase.y(options.ishoulderL) + personBase.y(options.ishoulderR) + personBase.y(options.ihipL) + personBase.y(options.ihipR)) / 4.0;
            
            mean(0+1).x = personBase.x(options.ihead);
            mean(0+1).y = personBase.y(options.ihead);
            
            scaleBody = sqrt((mean(1+1).x - mean(0+1).x)*(mean(1+1).x - mean(0+1).x) + (mean(1+1).y - mean(0+1).y)*(mean(1+1).y - mean(0+1).y));
        end
        
        for k = 1:options.numJointPairs1
            if (personBase.x(options.jointPairs1(k)) == -1.0 || ...
                    personBase.y(options.jointPairs1(k)) == -1.0)
                fvec = [fvec, -1.0*ones(1, options.numContactJointPairs0)];
            else
                for l = 1:options.numContactJointPairs0
                    if (personBase.x(options.contactJointPairs0(l)) == -1.0 || ...
                            personBase.y(options.contactJointPairs0(l)) == -1.0)
                        fvec = [fvec, -1.0];
                    else
                        x0 = personBase.x(options.jointPairs1(k));
                        y0 = personBase.y(options.jointPairs1(k));
                        
                        x0 = x0 - personBase.x(options.contactJointPairs0(l));
                        y0 = y0 - personBase.y(options.contactJointPairs0(l));
                        
                        x0 = x0/scaleBody;
                        y0 = y0/scaleBody;
                        
                        denom = 0.20;
                        if (options.isReal == 1)
                            denom = 0.8*denom;
                        end
                        fvec = [fvec, exp(-(x0*x0 + y0*y0) / denom)];
                    end
                end
                
            end
        end
        
        if ( options.numPeople > 1 )
            for k = 1:options.numJointPairs1
                if (personIJ.x(options.jointPairs1(k)) == -1.0 || ...
                        personIJ.y(options.jointPairs1(k)) == -1.0)
                    fvec = [fvec, -1.0*ones(1, options.numContactJointPairs0)];
                else
                    for l = 1:options.numContactJointPairs0
                        if (personBase.x(options.contactJointPairs0(l)) == -1.0 || ...
                                personBase.y(options.contactJointPairs0(l)) == -1.0)
                            fvec = [fvec, -1.0];
                        else
                            x0 = personIJ.x(options.jointPairs1(k));
                            y0 = personIJ.y(options.jointPairs1(k));
                            
                            x0 = x0 - personBase.x(options.contactJointPairs0(l));
                            y0 = y0 - personBase.y(options.contactJointPairs0(l));
                            
                            x0 = x0/scaleBody;
                            y0 = y0/scaleBody;
                            
                            denom = 0.20;
                            if (options.isReal == 1)
                                denom = 0.8*denom;
                            end
                            fvec = [fvec, exp(-(x0*x0 + y0*y0) / denom)];
                        end
                    end
                    
                end
            end
        end
    end
    contactFeatures(i, :) = fvec;
end

end
