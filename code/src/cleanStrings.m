function[Pclean, Rclean, Sclean] = cleanStrings(data)
    % Function to clean the strings for P, R, S
    % Eg. trimming off the strange spaces during python to matlab
    % conversions
    % 
    % [Pclean, Rclean, Sclean] = cleanStrings(data);
    %
    % Input: 
    % data = Data which is a cell, first element is a string triplet
    %        which needs to be trimmed
    %
    % Output:
    % Pclean = cleaned P labels
    % Rclean = cleaned R labels
    % Sclean = cleaned S labels
    %

    % Number of instances
    noInst = size(data, 1);
    
    % Cleaning the strings
    Pclean = cell(noInst, 1);
    Rclean = cell(noInst, 1);
    Sclean = cell(noInst, 1);
    for i = 1:noInst
        Pclean{i} = strtrim(data{i, 1}(1, :));
        Rclean{i} = strtrim(data{i, 1}(2, :));
        Sclean{i} = strtrim(data{i, 1}(3, :));
    end
end
