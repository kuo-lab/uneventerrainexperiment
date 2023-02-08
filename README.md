# README #

Humans plan for the near future to walk economically on uneven terrain
Osman Darici, Arthur D. Kuo
Faculty of Kinesiology and Biomedical Engineering Program, University of Calgary, Calgary, Alberta, Canada

Published in The Proceedings of the National Academy of Sciences, 2023

### What is this repository for? ###
This repository is for comparing Human and model data and running additional statistics.

### How do I run? ###

There are only two .m files to be run: plot_HumanModel and extratestsuneven. 

We recommend running extratestsuneven.m cell by cell.

This .m file basically gets the 8 terrains from the human and model structures first

Then scales them, and then runs the extra stats such as Bayes factors. 

Then using all speed data of the model and humans it finds the correlation coeffiecnts

Finally, it also get the model data of found by model predictive control and calculates again the correlation coefficients and plots as a funtion of horizon length.

You can select different model data to compare it with human data in the second cell using the flags:
optimalSolutions = 1;
constantTimeSolutions = 0;
noAnticipationSolutions = 0;


plot_HumanModel.m is simple and can be run to plot the human and model data for all terrains.

Data are stored in .mat format, which is a Matlab data file equivalent to HDF5.