% This function serves as a patch solution to deal with a bug in previous
% code that fails to save segment rejection flags.
% Running this function will replace all incorrect segments_rejFlag.mat
% with the correct ones in the intermediate3_segmented folder.
% Ran Xiao, 5/19/2021

% replace following two directories to your own
happe_directory_path = './happe-master/'; %replace with your HAPPE directory here
fpath_segmented = './demoData/intermediate3_segmented/';% replace with your directory for the intermediate3_segmented folder

% add EEGlab to path
eeglab_path = [happe_directory_path filesep 'Packages' filesep 'eeglab14_0_0b'];
% add HAPPE subfolders and EEGLAB plugin folders to path
addpath([happe_directory_path filesep 'acquisition_layout_information'],[happe_directory_path filesep 'scripts'],...
    eeglab_path,genpath([eeglab_path filesep 'functions']));

reject_min_amp = -40;
reject_max_amp = 40;
pipeline_visualizations_semiautomated = 0;

FileNames=dir(strcat(fpath_segmented,'*_segments_interp.set'));
FileNames={FileNames.name};

for current_file = 1:length(FileNames)
    EEG = pop_loadset('filename',FileNames{current_file},'filepath',fpath_segmented);
    EEG = eeg_checkset( EEG );

    EEG = pop_eegthresh(EEG,1,[1:EEG.nbchan] ,[reject_min_amp],[reject_max_amp],[EEG.xmin],[EEG.xmax],2,0);
                    EEG = pop_jointprob(EEG,1,[1:EEG.nbchan],3,3,pipeline_visualizations_semiautomated,...
                        0,pipeline_visualizations_semiautomated,[],pipeline_visualizations_semiautomated);
    EEG = eeg_rejsuperpose(EEG, 1, 0, 1, 1, 1, 1, 1, 1);
    rejFlag = EEG.reject.rejglobal;
    
    % Ran: save flags for whether each segment is rejected for
    % later analysis
    save([fpath_segmented strrep(FileNames{current_file}, '_segments_interp.set','_segments_rejectFlag.mat')],'rejFlag');
end
