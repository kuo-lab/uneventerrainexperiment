% Do extra analysis of statistics for complex uneven terrain paper,
% including Bayes factors.

% Data is assumed to consist of two main arrays:
% speedfluctuations(nconditions, nsteps, nsubjects) (trials averaged together)
% modelfluctuations(nconditions, nsteps)
%%
clear all; close all;
clc; beep off; 
load modelNominalVals
if ~exist('colororder') 
    colororder = [         0    0.4470    0.7410 
    0.8500    0.3250    0.0980 
    0.9290    0.6940    0.1250 
    0.4940    0.1840    0.5560 
    0.4660    0.6740    0.1880 
    0.3010    0.7450    0.9330 
    0.6350    0.0780    0.1840]; 
end 
%% Model IMU SPEED
%load All model data 
trialTypeStrs = {'oU',               'oD',              'UnD',             'oD2FFsBump',               'pyramid',              'longUnevenA',              'longUnevenB'};
% 'oD2FFsBump' refers to DnUP

optimalSolutions = 1;
constantTimeSolutions = 0;
noAnticipationSolutions = 0;

if optimalSolutions  
    trialTypeStrs2 = trialTypeStrs;
elseif constantTimeSolutions
    trialTypeStrs2 = {'oU_constantTime', 'oD_constantTime', 'UnD_constantTime', 'oD2FFsBump_constantTime', 'pyramid_constantTime', 'longUnevenA_constantTime', 'longUnevenB_constantTime'};
elseif noAnticipationSolutions
    trialTypeStrs2 = {'oU_no_anticipation', 'oD_no_anticipation', 'UnD_no_anticipation', 'oD2FFsBump_no_anticipation', 'pyramid_no_anticipation', 'longUnevenA_no_anticipation', 'longUnevenB_no_anticipation'};
end

len_trialTypeStrs = length(trialTypeStrs);
clear modelAllSpeeds

for kk = 1:len_trialTypeStrs    
    temp_optimaSolStrs = strcat('rw2midStance_Bf6Af6_', trialTypeStrs2{kk},'_HallwayExperimentApprox1_50');
    clear data; load (temp_optimaSolStrs);
    dataTemp = data.Bf6Af6;
    output = getSpeedWorkTimeGainRw2( dataTemp.xs,  dataTemp.ts,  dataTemp.indices,  dataTemp.energies,  dataTemp.optctrls,  dataTemp.bumps, nominalVals.nominalPushOffWorkRW, nominalVals.stepLenNomRW, nominalVals.stepTimeNomRW);   
    modelAllSpeeds.(trialTypeStrs{kk}) = output.speedImu(2:end);
end

modelAllSpeeds.longUnevenNormal =  ones(1, length(modelAllSpeeds.longUnevenA)) *  nominalVals.nominalStepSpeedRW;
modelAllSpeeds.simpleNormal =      ones(1, length(modelAllSpeeds.oU)) *  nominalVals.nominalStepSpeedRW;
modelAllSpeeds.pyramidNormal =     ones(1, length(modelAllSpeeds.pyramid)) *  nominalVals.nominalStepSpeedRW;
%% select the group
modelGr1 = {'simpleNormal', 'oU', 'oD', 'UnD',  'oD2FFsBump'  };
modelGr2 = {'longUnevenNormal', 'longUnevenA', 'longUnevenB' };
modelGr3 = { 'pyramidNormal', 'pyramid' };

modelAllGr = cat(2,modelGr1, modelGr2, modelGr3); modelGrnocontrol = cat(2,modelGr1(2:end),modelGr2(2:end),modelGr3(2:end));
modelgroups = {modelGr1, modelGr2, modelGr3}; 
ngroups = 3; 
% I want to try a different color for each condition, and
% a different marker for each subject
markers = 'o+*.xsd^v><ph'; % one for each subject
mycolors = [8 1 2 3 4; 8 5 6 8 8; 8 7 8 8 8 ]; % index by (group, conditionwithingroup)
colorlookup = colororder; 
colorlookup(8,:) = [1 1 1];
%% load human data
% close all
clear data dataOut dataModeltoPredict bumps dataMain
%we decided to run the corr. coef's with the actual simulation data no inital speed padded to the end and begining ------------OCT 18 2022
%so get rid of the first step of human data, so it will be 6 padding steps at the begining and then take whatever many steps the model simulation have
clear humanSpeedAll

load dataAvgStochasticNew
load dataStatsHallwayWalkingNew

humanSpeedAll.oU = dataStatsHallwayWalkingNew.oU_noConstraintWithLabel.meanSpeedsAll(2:end, :);
humanSpeedAll.oD = dataStatsHallwayWalkingNew.oD_noConstraintWithLabel.meanSpeedsAll(2:end,:);
humanSpeedAll.UnD = dataStatsHallwayWalkingNew.UnD_noConstraintWithLabel.meanSpeedsAll(2:end,:);
humanSpeedAll.oD2FFsBump = dataStatsHallwayWalkingNew.oD2FFsBump_noConstraintWithLabel.meanSpeedsAll(2:end,:);
humanSpeedAll.simpleNormal = dataStatsHallwayWalkingNew.Normal_NoConstraint.meanSpeedsofTestSubjects(2:end,:);

humanSpeedAll.longUnevenA = dataAvgStochasticNew.longUneven.longUnevenA.meanSpeedsAll(2:end,:);
humanSpeedAll.longUnevenB = dataAvgStochasticNew.longUneven.longUnevenB.meanSpeedsAll(2:end,:);
humanSpeedAll.longUnevenNormal = dataAvgStochasticNew.longUneven.Normal.meanSpeedsAll(2:end,:);

humanSpeedAll.pyramid = dataAvgStochasticNew.pyramid.pyramid.meanSpeedsAll(2:end,:);
humanSpeedAll.pyramidNormal = dataAvgStochasticNew.pyramid.Normal.meanSpeedsAll(2:end,:);

%just plot
for igroup = 1:ngroups
    modGr = modelgroups{igroup};
    nconditions = size(modGr,2);
    for tt = 1:nconditions  
        figure; hold on;
        f2 = plot(humanSpeedAll.(modGr{tt}), 'r'); plot(humanSpeedAll.(modGr{tt}), 'r.');    
        ylabel('speed'); xlabel('steps'); title(modGr{tt}) 
    end
end

%% Scaled correlations, where human & model are scaled to match
% and plotted against each other (all data). Produces human vs model plot.
% Also loads data by condition, for further analysis.
subplotmapping = [1 1 1 3 3; 4 4 4 0 0; 2 2 0 0 0];
%select trial group
figure; %hold on; axis equal; 
phs = [];% one plot for all conditions
for igroup = 1:ngroups
    modGr = modelgroups{igroup};
    % get scaled model flactuations for bayes
    nconditions = size(modGr,2);
    coefspeed = []; coefoffset = []; 
    for tt = 1:nconditions
        %figure; hold on; axis equal; % use this for one plot per condition
        subplot(2,2,subplotmapping(igroup,tt)); hold on; axis equal; 
        modelTemp = modelAllSpeeds.(modGr{tt})'; %so the model speed always resets
        humanTemp = humanSpeedAll.(modGr{tt}); % nsteps x nsubjects

        size_modelTemp = size(modelTemp,1); 
        size_humanTemp = size(humanTemp,1); 
        if size_humanTemp > size_modelTemp % if human has more steps, truncate human
            humanTemp = humanTemp(1:size_modelTemp,:); 
        elseif size_humanTemp < size_modelTemp % if model has more steps, truncate model
            modelTemp = modelTemp(1:size_humanTemp,:);
        else
        %
        end   
        Rall = []; R2all = []; RtempScaled = [];
        nsubjects = size(humanTemp,2); coefspeed = []; coefoffset = [];
        scaled_modelTemp = []; scaled_humanTemp = [];
        for k = 1:nsubjects      
            Rtemp = corr([humanTemp(:, k) modelTemp]);
            Rall = [Rall Rtemp(2)];
            [b,bint,r,rint,stats] = regress(humanTemp(:, k), [modelTemp ones(size_modelTemp,1)]); % scale model to human
            R2all = [R2all stats(1)];
            scaled_modelTemp = [scaled_modelTemp b(1)*modelTemp+b(2)];
            coefspeed(k) = b(1);   % store the coefficients just in case
            coefoffset(k) = b(2);

            [b,bint,r,rint,stats] = regress(modelTemp, [humanTemp(:,k) ones(size_modelTemp,1)]);  % scale human to model
            scaled_humanTemp = [scaled_humanTemp b(1)*humanTemp(:,k)+b(2)];

            %figure; hold on;
            %f1 = plot(humanTemp(:, k), 'r'); f2 = plot(scaled_modelTemp(:,end), 'g'); title(strcat('subj-', num2str(k), 'terrain-', modGr{tt}));
            %ylabel('speed'); xlabel('steps'); legend([f1 f2], 'human', 'scaledModel');
            if tt ~= 1, 
                 ph = plot(detrend(scaled_modelTemp(:,end),0), detrend(humanTemp(:, k),0), '.-', 'linewidth', 0.2, 'color', colorlookup(mycolors(igroup,tt),:)); 
    %            ph = plot(detrend(scaled_modelTemp(:,end),0), detrend(humanTemp(:, k),0), '.-', 'linewidth', 0.2, 'color', [0 1 0]);
            end
            xx = corr([humanTemp(:, k) scaled_modelTemp(:,end)]);
            RtempScaled = [RtempScaled xx(2)];
            %xx = corr([humanTemp(:, k) scaled_modelTemp(:,end)]);
            %RtempScaled = [RtempScaled xx(2)];

        end
        if tt ~= 1, phs = [phs ph]; end
        %plot(get(gca,'xlim'),get(gca,'ylim'),'--')
        modelAllSpeedsmatch.(modGr{tt}) = modelTemp; % these are the model & human
        humanAllSpeeds.(modGr{tt}) = humanTemp;      % with # of steps matched
        modelAllSpeedsScaled.(modGr{tt}) = scaled_modelTemp;
        humanAllSpeedsScaled.(modGr{tt}) = scaled_humanTemp;
        rhos.(modGr{tt}) = Rall;
        b1.(modGr{tt}) = coefspeed;
        b2.(modGr{tt}) = coefoffset;
    end % condition
end % groups
%xlabel('Model speed fluctuations (m/s)'); ylabel('Human speed fluctuations (m/s)');
%legend(phs, modelGrnocontrol)
linkaxes(get(gcf,'children'))
for i = get(gcf,'children')
    set(i, 'XAxisLocation', 'origin', 'yaxislocation', 'origin');
end

%examine the Rmeans of the simple regression
RmeanAll = [];
for igroup = 1:ngroups
    modGr = modelgroups{igroup};
    % get scaled model flactuations for bayes
    nconditions = size(modGr,2);
    for tt = 1:nconditions
        RmeanAll.(modGr{tt}) =  mean(rhos.(modGr{tt}));
    end
end
RmeanAll

% examine the b's (not that informative)
for icond = 1:length(modelAllGr)
    thiscond = modelAllGr{icond};
    b1means.(thiscond) = mean(b1.(thiscond));
    b2means.(thiscond) = mean(b2.(thiscond));
end
%% calculates the bayes factor
% now for each condition and each step, test whether scaled model predicts
% human. Test all subjects against their corresponding scaled model
% close all
nshuffles = 1000; clf; hold on;
for igroup = 1:ngroups
    modGr = modelgroups{igroup};
    nconditions = size(modGr,2);
    
    for tt = 1:nconditions   % all conditions, all terrains

        modelTemp = modelAllSpeedsmatch.(modGr{tt}); %load the scaled model, nsteps by nsubjects
        humanTemp = humanAllSpeedsScaled.(modGr{tt}); % & humans w/ matched steps

        nsteps = size(humanTemp,1); % humanTemp is nsteps x nsubjects
        pmodelTemp = [];
        pnullTemp = [];
        permutesteps = randperm(nsteps);
        for i=1:nshuffles, 
            permutesteps2(i,1:nsteps) = randperm(nsteps); 
        end
        allpnulls=zeros(nsteps,nshuffles);
        for j = 1:nsteps%        nsteps is a number   nsteps(i) % all steps of each condition
              %figure; hold on;
            data = humanTemp(j,:); % all subjects (averaged across their trials)%<-----------------jth step of every subject
            [h,p] = ttest(data - modelTemp(j,:)); % model hypothesis test %<-----------------jth step of every subject
    %         subplot(2,1,1); hold on; plot(data, 'k'); plot(modelTemp(j,:), 'r', 'markersize', 15); ylabel('data-model')
            pmodelTemp(j,:) = p;
            data = humanTemp(permutesteps(j),:);
            [h,p] = ttest(data - modelTemp(j,:)); % null hypothesis test, using shuffled human againts model 
            pnullTemp(j,:) = p;      

            for i = 1:nshuffles
                data = humanTemp(permutesteps2(i,j),:); 
                [h,p] = ttest(data - modelTemp(j,:));
                allpnulls(j,i) = p;
            end
    %         subplot(2,1,2); hold on; plot(data, 'k'); plot(data, 'k.', 'markersize', 15); ylabel('data')
        end
        nstepscond.(modGr{tt}) = nsteps;
    %     subplot(3,1,3); 
    %     figure
    %     hold on; f1 = plot(pmodelTemp, 'r');  plot(pmodelTemp, 'r.', 'markersize', 15); 
    %      f2 = plot(pnullTemp, 'g');   plot(pnullTemp, 'g.', 'markersize', 15);
    %     title(strcat('terrain-', modGr{tt})); ylabel('Pval'); xlabel('steps'); legend([f1 f2], 'model', 'null');

        % find odds ratios for each terrain (condition)
        logpmodels = sum(log(pmodelTemp'),2); % log of product of probabilities across all steps, also called log likelihood
        % above yields a vector with log p's one for each terrain
        logpnulls = sum(log(pnullTemp'),2);   % log of product of probabilities across all steps
        logallpnulls = sum(log(allpnulls));
        oddsratiosall = exp(logpmodels - logallpnulls);
        logoddsTemp = logpmodels - logpnulls;  % this is log of ratio of probabilities
        oddsratiosTemp = exp(logoddsTemp);       % convert back from log to get true odds (could be wrong)

        pmodel.(modGr{tt}) = pmodelTemp'; % probability of human matching model with t distribution
        pnull.(modGr{tt}) = pnullTemp';   % probability of shuffled human matching model
        meanlogoddsratiosall.(modGr{tt}) = [mean(logpmodels-logallpnulls) std(logpmodels-logallpnulls)];
        if tt ~= 1, hist(logpmodels-logallpnulls); end; 
    end % conditions
end % ngroups


% bayesinformationcriterion = log(nconditions*nsteps*nsubjects+2*nsubjects) - 2*sum(logpmodels);
% the 2*nsubjects is for the two coefficients we fit for each subject

% bits per step for the different terrains
clear bits
for icond = 1:length(modelAllGr)
    thiscond = modelAllGr{icond};
    bits(icond,:) = meanlogoddsratiosall.(thiscond)/log(2)/nstepscond.(thiscond);
end
mean(bits([2 3 4 5 7 8 10],:))
meanlogoddsratiosall

crap = struct2cell(meanlogoddsratiosall);
crap = [crap{:}];
bayesfactors = exp(crap(1:2:end)); 
[x,i] = min(bayesfactors([2 3 4 5 7 8 10])) 
% total number of steps
crap = cell2mat(struct2cell(nstepscond)); sum(crap([2 3 4 5 7 8 10])) % 133
%% Big correlations, all steps all subjects for each condition
% where we do not account for scale, yet still end up with pretty
% good correlations. This yields good p-values and tight CIs.
% Let's use this for the finite horizon correlation plot.
clear rhos rhosraw pvals pvalsraw models human
for igroup = 1:ngroups
    modGr = modelgroups{igroup};
    for tt = 1:length(modGr)
        nsubjects = size(humanAllSpeeds.(modGr{tt}),2); 
        model = []; modelraw = [];
        human = [];
        model = [model; reshape(detrend(modelAllSpeedsScaled.(modGr{tt}),0),[],1)];
        modelTemp = repmat(modelAllSpeedsmatch.(modGr{tt}),1,nsubjects);
        modelraw = [modelraw; reshape(detrend(modelTemp,0),[],1)]; 
        human = [human; reshape(detrend(humanAllSpeeds.(modGr{tt}),0),[],1)];
        models.(modGr{tt}) = model;
        modelsraw.(modGr{tt}) = modelraw;
        humans.(modGr{tt}) = human;
        [rho,pval,rl,ru] = corrcoef(model(:), human(:)); % rl & ru are 95% CI
        rhos.(modGr{tt}) = [rho(1,2) rho(1,2)-rl(1,2)]; % rho ± half CI
        pvals.(modGr{tt}) = pval(1,2);
        [rho,pval,rl,ru] = corrcoef(modelraw(:), human(:));
        rhosraw.(modGr{tt}) = [rho(1,2) rho(1,2)-rl(1,2)];
        pvalsraw.(modGr{tt}) = pval(1,2);
    end
end


crapScaled = cell2mat(struct2cell(rhos)); crapScaled = crapScaled([2 3 4 5 7 8 10],1);
fprintf(1,'SCALED mean ± sd across terrains = %f, %f\n', mean(crapScaled), std(crapScaled))

crap = cell2mat(struct2cell(rhosraw)); crap = crap([2 3 4 5 7 8 10],1);
fprintf(1,'mean ± sd across terrains = %f, %f\n', mean(crap), std(crap))

% correlation of all data from all terrains and subjects at once
conds = [2 3 10 4 5 7 8]; % U D P UD D&UD C1 C2
crap = struct2cell(models); bigmodels = cat(1,crap{conds});
crap = struct2cell(humans); bighumans = cat(1,crap{conds});
crap = struct2cell(modelsraw); bigmodelsraw = cat(1,crap{conds});
[rho, pval] = corrcoef(bigmodels, bighumans); bigrho=rho(1,2); bigpval=pval(1,2);
[rho, pval] = corrcoef(bigmodelsraw, bighumans); bigrhoraw=rho(1,2); bigpvalraw=pval(1,2);


% make a bar graph with CI, in paper's condition order
crap = struct2cell(rhosraw); correlations = cat(1,crap{:});
clf; subplot(121); b=bar(correlations(conds,1), 0.2); hold on;
er=errorbar(1:length(conds), correlations(conds,1), [], correlations(conds,2));
er.Color = [0 0 0]; er.LineStyle = 'none';
b.FaceColor = 'flat'; b.CData = colororder;
set(gca, 'xticklabels', {'U','D','P','UD', 'D&UD', 'C1', 'C2'});
meancorr = mean(correlations(conds,1)); % 0.5528 raw; 0.5620 full
plot(get(gca,'xlim'), meancorr*[1 1], ':')
ylabel('Correlation'); set(gca,'ylim', [0 0.8]);

% Also tried a single correlation based on all steps of all conditions,
% which yielded onerho = 0.5356, onepval = 2.5e-112 to machine precision. Not using because
% it is biased towards more steps.
% using unscaled data, onerhoraw = 0.4838, onepval = 5.2e-89
%[rho, pval] = corrcoef(allmodels, allhumans); onerho=rho(1,2); onepval=pval(1,2);
%[rho, pval] = corrcoef(allmodelsraw, allhumans); onerhoraw=rho(1,2); onepvalraw=pval(1,2);
%% MPC correlation coef.s BIG correlation

colors = jet(7);
clear corrMPC;

load('corrMPC_6stepsPaddingHr.mat') ; corrMPC = corrMPC_6stepsPaddingHr; clear corrMPC_6stepsPaddingHr; steps3padding = 0;
%load('corrMPC_3stepsPaddingHr.mat') ; corrMPC = corrMPC_3stepsPaddingHr; clear corrMPC_3stepsPaddingHr; steps3padding = 1;

subplot(122); hold on; flegends = [];

modelGr1 = {'oU', 'oD', 'UnD',  'oD2FFsBump'  };
modelGr2 = {'longUnevenA', 'longUnevenB' };%, 
modelGr3 = { 'pyramid' };
modelgroups = {modelGr1, modelGr2, modelGr3}; 
trialTypeStrForLegend = {'oU', 'oD', 'UnD',  'DnUP', 'comp1',  'comp2', 'pyramid'}; 
figInd = 1;
crap = cell2mat(struct2cell(corrMPC)); maxhorlen = max(cat(2,crap.horLenAll)); extrhosraw = [];
clear rhos rhosraw pvals pvalsraw models human
for igroup = 1:ngroups
    modGr = modelgroups{igroup};
   
    for tt = 1:length(modGr)
        nsubjects = size(humanAllSpeeds.(modGr{tt}),2); 
           
        horLenAll = corrMPC.(modGr{tt}).horLenAll; 
        sizehorLenAll = length(horLenAll);
        
        rhos.(modGr{tt}) = [];
        rhosCI.(modGr{tt}) = [];
        pvals.(modGr{tt}) = [];
        rhosraw.(modGr{tt})= [];
        rhosrawCI.(modGr{tt})= [];
        pvalsraw.(modGr{tt})= [];
        
        for numHrz = 1:sizehorLenAll
                      
            model = []; modelraw = []; human = [];     
            strTmp = strcat('hrz', num2str(horLenAll(numHrz)));
            dataHrz = corrMPC.(modGr{tt}).corrMPCvsHuman.(strTmp);      
            model = [model; reshape(detrend(dataHrz.scaledModelSpeed,0),[],1)];
            modelTemp = repmat(dataHrz.modelSpeed',1,nsubjects);       
            modelraw = [modelraw; reshape(detrend(modelTemp,0),[],1)];     
            if steps3padding
                lenModel = length(dataHrz.modelSpeed);               
                %we need to truncate the human take out first 3 steps, so it becomes 3 steps paading and them take as much as model step num
                tempHuman = humanAllSpeeds.(modGr{tt})(4:end,:);
                tempHuman = tempHuman(1:lenModel,:);                
                human = [human; reshape(detrend(tempHuman,0),[],1)];
            else
                human = [human; reshape(detrend(humanAllSpeeds.(modGr{tt}),0),[],1)];
            end
            
            % models.(modGr{tt}) = model;
            % modelsraw.(modGr{tt}) = modelraw;
            % humans.(modGr{tt}) = human;
            
            [rho,pval,rl,ru] = corrcoef(model(:), human(:)); % rl & ru are 95% CI
            rhos.(modGr{tt}) =   [rhos.(modGr{tt})   rho(1,2) ]; % rho ± half CI
            rhosCI.(modGr{tt}) = [rhosCI.(modGr{tt}) rho(1,2)-rl(1,2)];
            pvals.(modGr{tt}) =  [pvals.(modGr{tt})  pval(1,2)];
            
            [rho,pval,rl,ru] = corrcoef(modelraw(:), human(:));
            rhosraw.(modGr{tt}) =   [rhosraw.(modGr{tt})     rho(1,2)];
            rhosrawCI.(modGr{tt}) = [rhosrawCI.(modGr{tt})   rho(1,2)-rl(1,2)];
            pvalsraw.(modGr{tt}) =  [pvalsraw.(modGr{tt})    pval(1,2)];
        end     
%          plot(horLenAll, rhos.(modGr{tt}),'linewidth',2); plot(horLenAll, rhos.(modGr{tt}), '.', 'Markersize', 20); 
        exthorlen = [horLenAll horLenAll(end)+1:maxhorlen]; 
        extrhosraw = [extrhosraw; [rhosraw.(modGr{tt}) repmat(rhosraw.(modGr{tt})(end),1,maxhorlen-horLenAll(end))]];
        %flegends(figInd) = plot(horLenAll, rhosraw.(modGr{tt}), 'linewidth',2,  'color', colors(tt,:)); plot(horLenAll, rhosraw.(modGr{tt}), '.', 'Markersize', 20,  'color', colors(tt,:)); % plot correlations for one terrain
         figInd =  figInd + 1;
    end % condition
end % group 
plot(exthorlen, extrhosraw, 'linewidth',2); plot(exthorlen, extrhosraw, '.', 'Markersize', 10); % plot correlations for one terrain
legend(trialTypeStrForLegend); xlim([0 30]); ylabel('corr. coef (big correlation)'); xlabel('horizon step number')
plot(exthorlen, mean(extrhosraw),'color', [0.5 0.5 0.5], 'linewidth',3, 'displayname', 'mean')
%errorbar(exthorlen, mean(extrhosraw), std(extrhosraw),[])
set(gca, 'ylim', [0 0.8]);
% 6-step padding 
sixstepmean = mean(extrhosraw); % 
