
% Plots human subject average speeds and model speed per terrain
% Osman Darici 2022

clear all; close all

load modelData_PNAS
load humanData_PNAS

conditions = {'controlSimple', 'oU', 'oD', 'UnD', 'DnUD', 'pyramid', 'C1', 'C2'};
numSubjects = 12;

figure; hold on
figIndHuman = 4;
figIndModel = 0;

for z = 1:length(conditions)
    
    subplot(4,4, figIndModel+z); hold on
    speedConditionModel = modelData_PNAS.speed.(conditions{z});
    timeConditionModel = modelData_PNAS.time.(conditions{z});
    plot(timeConditionModel, speedConditionModel, 'linewidth', 1); plot(timeConditionModel, speedConditionModel, '.', 'markersize', 10)
    plot(mean(timeConditionModel,2), mean(speedConditionModel, 2), 'r', 'linewidth', 2); plot(mean(timeConditionModel,2), mean(speedConditionModel, 2), 'r.' , 'markersize', 15);
    plot([0 0], ylim, 'k--'); ylabel('speed (dim.less)'); xlabel('time(s)'); title(strcat(conditions{z}, '-Model'));
    
    subplot(4,4,figIndHuman+z); hold on
    speedCondition = humanData_PNAS.speed.(conditions{z});
    timeCondition = humanData_PNAS.time.(conditions{z});  
    plot(timeCondition, speedCondition, 'linewidth', 1); plot(timeCondition, speedCondition, '.', 'markersize', 10)
    plot(mean(timeCondition,2), mean(speedCondition, 2), 'r', 'linewidth', 3); plot(mean(timeCondition,2), mean(speedCondition, 2), 'r.' , 'markersize', 15);
    plot([0 0], ylim, 'k--'); ylabel('speed (m/s)'); xlabel('time(s)'); title(strcat(conditions{z}, '-Human'));
    
    if z == 4
        figIndHuman = 8;
        figIndModel = 4;
    end
end