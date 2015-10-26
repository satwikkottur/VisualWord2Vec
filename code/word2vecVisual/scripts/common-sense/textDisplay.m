x = 1:10; y = 1:10; scatter(x,y);
a = [1:10]'; b = num2str(a); c = cellstr(b);
dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
text(x+dx, y+dy, c);
