# HAPPE4InfantMotorStudy
Ran Xiao, ran.xiao@duke.edu Duke University, April 2021

Adaptation of HAPPE pipeline to Biosemi EEG data from the Infant motor study. 

Prerequisit: HAPPE Pipeline, https://github.com/lcnhappe/happe
Gabard-Durnam et al., 2018; https://doi.org/10.3389/fnins.2018.00097

When first time using HAPPE
1. Download the HAPPE toolbox into your working directory
2. Run EEGLab with Matlab commandline first and be sure to install the biosemi plugin for the EEGlab.
3. Make sure to change Memory Allocation Setting in EEGLAB the same as the one listed in HAPPE readme file, otherwise memory error will occur.
4. Copy Event time CSV files into the working direcotry, or change LatencyDir to correponding one. 
5. Copy all .bdf files to be processed in one folder under working diretory.
6. make changes of directory variables in the ProprocessingWithHAPPE.m code.

Bugs found in HAPPE and fixed in this toolbox:
1. EGIload prepopulate unrealisticly large matrix, generating errors when running the EGI demo.
2. (FIXED) segment_data and segment_interpolation are treated separately, and an error can arise when users choose not to segment data but selected segment interpolation
3. (FIXED) segment_rejection flag is never used, meaning no matter what users choose, the pipeline will always reject bad segment
4. (ADDED) added a feature that saves a matrix that indicates whether each segment is rejected so that time info can be recovered later.
5. (ADDED) added try-catch mechanism to handle challenging sessions
6. 
