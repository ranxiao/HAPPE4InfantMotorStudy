# HAPPE4InfantMotorStudy
Ran Xiao, ran.xiao@duke.edu Duke University, April 2021

Adaptation of HAPPE pipeline to Biosemi EEG data from the Infant motor study. 

Prerequisit: HAPPE Pipeline, https://github.com/lcnhappe/happe
Gabard-Durnam et al., 2018; https://doi.org/10.3389/fnins.2018.00097

When first time using HAPPE
1. Download the HAPPE toolbox into your working directory
2. Run EEGLab with commandline first and be sure to install the biosemi plugin for the EEGlab.
3. Make sure to change Memory Allocation Setting in EEGLAB the same as the one listed in HAPPE readme file, otherwise memory error will occur.
4. Copy Event time CSV files into the working direcotry, or change LatencyDir to correponding one. 
5. make changes of working directory in the ProprocessingWithHAPPE.m code to your local directory.

