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

%% Compute Global Features
function globalFeatures = ComputeGlobalPositionFeatures(options, inputFile, partXY, visualizeFeature, isReal)

if (visualizeFeature ~= 0)
    fid = fopen(inputFile);
    colsAsCells = textscan(fid, '%d%s%s', 'Delimiter', ',');
    fclose(fid);
    labels = double(colsAsCells{1});
    filenames = colsAsCells{2};
end

if ( isReal ~= 0 )
    % filepath   = '../../data/output/real_poses/full_exp_1/avgPoseImgs/';
    filepath   = '../../data/output/image_collection/full_exp_1/foundImgs/';
    outputPath = '../../data/output/calculate_real_features/full_exp_1/features/globalFeatImgs/';
else
    %filepath   = '../../data/output/clipart_collection/full_exp_1/renderedImgs/';
    %outputPath = '../../data/output/clipart_collection/full_exp_1/features/globalFeatImgs/';
end

if ( options.numPeople == 1 )
    globalFeatures = zeros(options.numImgs, options.numPeople*(options.numJointPairs)*options.numGaussians);
else
    globalFeatures = zeros(options.numImgs, options.numPeople*(2*options.numJointPairs)*options.numGaussians);
end

for i = 1:options.numImgs
    if (visualizeFeature ~= 0)
        fullFilename = filenames{i};
        fprintf('i=%d, file=%s\n', i, fullFilename);
        parts = strsplit(fullFilename, '.');
        name = parts{1};
        ext = parts{2};
    end
    
    fvec = [];
    
    for j = 1:options.numPeople
        
        if (visualizeFeature ~= 0)
            %         realFilename = sprintf('%s_avgPoses_%02d.png', name, j);
            realFilename = fullFilename;
            
            if ( strcmp( lower(ext), 'gif') == 0 )
                inImg = imread( fullfile(filepath, realFilename) );
            else
                try
                    [inImg, inImgMap] = imread( fullfile(filepath, realFilename), 1 );
                catch
                    [inImg, inImgMap] = imread( fullfile(filepath, realFilename));
                end
                
                if (ndims(inImg) < 3)
                    inImg = ind2rgb(inImg, inImgMap);
                end
            end
            
            inImgSize = size(inImg);
            if ( length(inImgSize) == 2 )
                temp = [];
                temp(:, :, 1) = inImg;
                temp(:, :, 2) = inImg;
                temp(:, :, 3) = inImg;
                inImg = temp;
            end
            
            inImgSize = size(inImg);
            numR = inImgSize(1);
            numC = inImgSize(2);
            
            outImg = inImg;
        end
        otherPerson = mod( j, options.numPeople )+1;
        mean = [struct('x', 'y'); struct('x', 'y')];
        
        personIJ = partXY{i, j};
        personOther = partXY{i, otherPerson};
        
        if (personIJ.x(options.ihead) == -1.0 || ...
                personIJ.y(options.ihead) == -1.0 || ...
                personIJ.x(options.ishoulderL) == -1.0 || ...
                personIJ.y(options.ishoulderL) == -1.0 || ...
                personIJ.x(options.ishoulderR) == -1.0 || ...
                personIJ.y(options.ishoulderR) == -1.0 || ...
                personIJ.x(options.ihipL) == -1.0 || ...
                personIJ.y(options.ihipL) == -1.0 || ...
                personIJ.x(options.ihipR) == -1.0 || ...
                personIJ.y(options.ihipR) == -1.0 ...
                )
            
            fvec = [fvec, -1.0*ones(1, options.numGaussians*2*options.numJointPairs)];
            if (visualizeFeature ~= 0)
                imwrite(outImg, fullfile(outputPath, sprintf('%s_%02d.jpg', name, j)) );
            end
        else
            
            mean(1+1).x = (personIJ.x(options.ishoulderL) + personIJ.x(options.ishoulderR) + personIJ.x(options.ihipL) + personIJ.x(options.ihipR)) / 4.0;
            mean(1+1).y = (personIJ.y(options.ishoulderL) + personIJ.y(options.ishoulderR) + personIJ.y(options.ihipL) + personIJ.y(options.ihipR)) / 4.0;
            
            mean(0+1).x = personIJ.x(options.ihead);
            mean(0+1).y = personIJ.y(options.ihead);
            
            orienBody = -atan2(mean(0+1).y - mean(1+1).y,  mean(0+1).x - mean(1+1).x) + pi/2.0;
            scaleBody =  sqrt((mean(1+1).x - mean(0+1).x)*(mean(1+1).x - mean(0+1).x) + (mean(1+1).y - mean(0+1).y)*(mean(1+1).y - mean(0+1).y));
            
            if (visualizeFeature ~= 0)
                dx = 30.0*cos(orienBody);
                dy = 30.0*sin(orienBody);
                
                %                 std::VtDrawLine(inImg, midShoulder.x, midShoulder.y, midShoulder.x + dx, midShoulder.y + dy, (BYTE *)&clr);
            end
            
            if (visualizeFeature ~= 0)
                
                c = linspace(1, numC, numC);
                r = linspace(1, numR, numR);
                [C, R] = meshgrid(c, r);
                
                x0 = C;
                y0 = R;
                
                
                x0 = x0 - mean(1+1).x;
                y0 = y0 - mean(1+1).y;
                
                x0 = x0./scaleBody;
                y0 = y0./scaleBody;
                
                x1 = x0.*cos(orienBody) - y0.*sin(orienBody);
                y1 = x0.*sin(orienBody) + y0.*cos(orienBody);
                x1 = x1./2.0;
                
                prob = 0.0;
                
                denom = 0.5;
                if (options.isReal == 1)
                    denom = 0.8*denom;
                end
                denom = 0.75;
                
                prob = max(prob, exp(-(x1.*x1 + y1.*y1) ./ (denom)));
                prob = max(prob, exp(-(x1.*x1 + (y1 - 1.0).*(y1 - 1.0)) ./ (denom)));
                prob = max(prob, exp(-(x1.*x1 + (y1 + 1.0).*(y1 + 1.0)) ./ (denom)));
                
                outImg(:, :, 1) = uint8(255.0*prob);
                outImg(:, :, 2) = uint8(y1 > 0.0).*uint8(255.0*prob) + uint8(y1 <= 0.0).*uint8(inImg(:, :, 2));
                
                %                 fullfile(outputPath, sprintf('%s_%02d.jpg', name, j))
                if (visualizeFeature ~= 0)
                    imwrite(outImg, fullfile(outputPath, sprintf('%s_%02d.jpg', name, j)) );
                end
            end
            
            for k = 1:options.numJointPairs
                
                if ( personIJ.x(options.jointPairs1(k)) == -1.0 || ...
                        personIJ.y(options.jointPairs1(k)) == -1.0)
                    fvec = [fvec, -1.0*ones(1, options.numGaussians)];
                else
                    x0 = personIJ.x(options.jointPairs1(k));
                    y0 = personIJ.y(options.jointPairs1(k));
                    
                    x0 = x0 - mean(1+1).x;
                    y0 = y0 - mean(1+1).y;
                    
                    x0 = x0/scaleBody;
                    y0 = y0/scaleBody;
                    
                    x1 = x0*cos(orienBody) - y0*sin(orienBody);
                    y1 = x0*sin(orienBody) + y0*cos(orienBody);
                    x1 = x1/2.0;
                    
                    denom = 0.5;
                    if (options.isReal == 1)
                        denom = 0.8*denom;
                    end
                    
                    fvec = [fvec, exp(-(x1*x1 + y1*y1) / (denom)), ...
                        exp(-(x1*x1 + (y1 - 1.0)*(y1 - 1.0)) / (denom)), ...
                        exp(-(x1*x1 + (y1 + 1.0)*(y1 + 1.0)) / (denom))];
                end
                
                if ( options.numPeople > 1 )
                
                    if (personOther.x(options.jointPairs1(k)) == -1.0 || ...
                            personOther.y(options.jointPairs1(k)) == -1.0 )
                        fvec = [fvec, -1.0*ones(1, options.numGaussians)];
                    else
                        x0 = personOther.x(options.jointPairs1(k));
                        y0 = personOther.y(options.jointPairs1(k));

                        x0 = x0 - mean(1+1).x;
                        y0 = y0 - mean(1+1).y;

                        x0 = x0/scaleBody;
                        y0 = y0/scaleBody;

                        x1 = x0*cos(orienBody) - y0*sin(orienBody);
                        y1 = x0*sin(orienBody) + y0*cos(orienBody);
                        x1 = x1/2.0;

                        denom = 0.50;
                        if (options.isReal == 1)
                            denom = 0.8*denom;
                        end
                        fvec = [fvec, exp(-(x1*x1 + y1*y1) / (denom)), ...
                            exp(-(x1*x1 + (y1 - 1.0)*(y1 - 1.0)) / (denom)), ...
                            exp(-(x1*x1 + (y1 + 1.0)*(y1 + 1.0)) / (denom))];
                    end
                end
            end
        end
    end
    globalFeatures(i, :) = fvec;
end

end