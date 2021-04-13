% ActivityTimePull
% V1. Jeff Nishida
% 5/20/15
% V2. Ran Xiao
% 12/9/2016

% In V2, it is revised to accomodate new csv data format.
% Pulls time of all activities from csv and return 28 time points in the following order:
% (1)baseline 1 start	(2)baseline 1 stop	(3)baseline 1 20 s start (4)baseline 1 20 s end	
% (5)toy 1 start	(6)toy 1 stop (7)no toy 1 start	(8)no toy 1 stop  (9)toy 2 start	(10)toy 2 stop	
% (11)no toy 2 start	(12)no toy 2 stop	(13)toy 3 start	(14)toy 3 stop (15)no toy 3 start	(16)no toy 3 stop	
% (17)toy 4 start	(18)toy 4 stop	(19)no toy 4 start	(20)no toy 4 stop	(21)toy 5 start	(22)toy 5 stop
% (23)no toy 5 start (24)no toy 5 stop	
% (25)baseline 2 start	(26)baseline 2 stop	(27)baseline 2 20 s start	(28)baseline 2 20 s end

% csvPath = full path to csv file


function times = ActivityTimePull(csvPath)

    fid = fopen(csvPath,'r');

    C = textscan(fid,'%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s','delimiter',',','headerlines',0);

    for n = 1:length(C)
        if ischar(C{n}) || iscell(C{n})
            C{n} = regexprep(C{n},'\t','');
            C{n} = regexprep(C{n},' ','');
        end
    end

    times = zeros(1,length(C)-3);
    for i = 4:length(C)
        times(i-3) = cellfun(@MM_SStoSeconds,C{i}(2));
    end
end

% Local Functions
function sec = MM_SStoSeconds(str)
    % Converts MM:SS string format to seconds number
    idxCol = regexp(str,':');

    if isempty(idxCol)
        sec = NaN;
    elseif idxCol == 1
        % No colon or colon is first character. All numbers are seconds
        idxNum = regexp(str,'\d');
        sec = str2double(str(idxNum));    
    else
        sec = 60*str2double(str(1:idxCol-1))+str2double(str(idxCol+1:end));
    end
end