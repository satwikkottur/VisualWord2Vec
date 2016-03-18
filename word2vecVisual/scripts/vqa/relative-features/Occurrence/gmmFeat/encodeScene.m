function [fv,parts] = encodeScene(S, GAbsPos, GRelPos, scene, objectId)
    %Take a scene and GMMs and encode the object with respect to the scene.
    
    objectIndex = find(scene.objectIds == objectId);
    numAttributes = cellfun(@(x)(numel(x)),S.attributeCountList);
    numAttributeIndicators = cellfun(@(x)(sum(x)),S.attributeCountList);
    
    if numel(objectIndex)
        %get the important stuff out
        objX = scene.attributes{objectIndex}(S.X_INDEX);
        objY = scene.attributes{objectIndex}(S.Y_INDEX);
        objZ = scene.attributes{objectIndex}(S.Z_INDEX);
        flip = scene.attributes{objectIndex}(S.FLIP_INDEX);
        scales = getCanonicalScales();
        xMultiplier = getFlipMultiplier(flip)/scales(objZ+1);
        yMultiplier = abs(xMultiplier);
        %global component
        objV = [objX, objY];
        %fvGlobal = [gmmEncodeF(objV,GAbsPos); 0; objZ; flip; scene.attributes{objectIndex}(S.EXTRA_INDEX:end)'];
        attribs = scene.attributes{objectIndex}(S.EXTRA_INDEX:end);
        binaryAttribs = encodeAttributesBinary(attribs,S.attributeCountList{objectIdToIndex(objectId)});
        assert(numel(binaryAttribs) == numAttributeIndicators(objectIdToIndex(objectId)));
        
        fvGlobal = [gmmEncodeF(objV,GAbsPos); 0; objZ; flip; binaryAttribs];
    else
        objV = getSceneCenter();
        objX = objV(1); objY = objV(2);
        %fvGlobal = [zeros(GAbsPos.NComponents,1); 1; 0; 0; zeros(numAttributes(objectIdToIndex(objectId)),1)];
        fvGlobal = [zeros(GAbsPos.NComponents,1); 1; 0; 0; zeros(numAttributeIndicators(objectIdToIndex(objectId)),1)];
        xMultiplier = 1; 
        yMultiplier = 1;
    end

    
    %the number of dimensions per object (each relative component +
    %a presence indicator + z + flip + the attributes)
    %perObjectFeatureCount = numAttributes+GRelPos.NComponents+3;
    perObjectFeatureCount = numAttributeIndicators+GRelPos.NComponents+3;
    %the object goes from Start(i):End(i) INCLUSIVE
    perObjectEnd = cumsum(perObjectFeatureCount);
    perObjectStart = [1,perObjectEnd(1:end-1)+1];
        
    fvRelative = nan(sum(perObjectFeatureCount),1);
    partsRelative = nan(sum(perObjectFeatureCount),1);
    
    for j=1:numel(S.objectList)
        fvToAdd = [];
        otherIndex = find(scene.objectIds == indexToObjectId(j));
        if(numel(otherIndex))
            otherX = scene.attributes{otherIndex}(S.X_INDEX);
            otherY = scene.attributes{otherIndex}(S.Y_INDEX);
            otherZ = scene.attributes{otherIndex}(S.Z_INDEX);
            otherFlip = scene.attributes{otherIndex}(S.FLIP_INDEX);
            attribs = scene.attributes{otherIndex}(S.EXTRA_INDEX:end);
            binaryAttribs = encodeAttributesBinary(attribs,S.attributeCountList{j});
            relativeVector = [objX - otherX, objY - otherY];
            relativeVector = [xMultiplier*relativeVector(1), yMultiplier*relativeVector(2)];
            assert(numel(binaryAttribs) == numAttributeIndicators(j));
            fvToAdd = [gmmEncodeF(relativeVector,GRelPos); 0; otherZ; otherFlip; binaryAttribs];         
        else
            fvToAdd = [zeros(GRelPos.NComponents,1); 1; 0; 0; zeros(numAttributeIndicators(j),1)];
        end
        fvRelative(perObjectStart(j):perObjectEnd(j)) = fvToAdd;
        partsRelative(perObjectStart(j):perObjectEnd(j)) = j;
    end
    assert(sum(isnan(fvRelative)) == 0);
    fv = [fvRelative; fvGlobal];
    parts = [partsRelative; -1*ones(size(fvGlobal))];
end