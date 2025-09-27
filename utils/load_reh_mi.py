import json
from mne.io import read_raw_edf
from dateutil.parser import parse
import glob as glob
from datetime import datetime
import numpy as np
import os

DATA_PATH = os.path.expanduser('~') + '/datasets/CS2R MI EEG dataset/'

def load_REHMI(subject, training, preprocessing_dict: dict,
                      classes_labels =  ['Fingers', 'Wrist','Elbow','Rest'], 
                      mi_duration = 4, valid_subs = True):
    """
    Load training/testing EEG data for a specific subject from the REHMI MI dataset.

    Parameters
    ----------
    subject_index : int
        Index of the subject in the subject list (not the subject ID).
    training : bool
        True for training session, False for testing session.
    preprocessing_dict : dict
        Dictionary with 'low_cut', 'high_cut', and 'sfreq'.
    classes_labels : list
        Classes to include.
    mi_duration : int
        Duration (seconds) of MI segment.
    valid_subs : bool
        Whether to use only valid subjects.

    Returns
    -------
    data : np.ndarray
    labels : np.ndarray
    onset : np.ndarray
    duration : np.ndarray
    description : np.ndarray
    """

    subs_dir = "valid_subjects/" if valid_subs else "all_subjects/"
    data_path = os.path.join(DATA_PATH, subs_dir)

    session = 1 if training else 2
    
    # Get all subjects files with .edf format.
    subjectFiles = glob.glob(data_path + 'S_*/')
    
    # Get all subjects numbers sorted without duplicates.
    subjectNo = list(dict.fromkeys(sorted([x[len(x)-4:len(x)-1] for x in subjectFiles])))
    # print(SubjectNo[subject].zfill(3))
       
    num_runs = 5
    sfreq = preprocessing_dict.get("sfreq", 128.0)
    mi_duration = mi_duration #4.5

    data = np.zeros([num_runs*51, 32, int(mi_duration*sfreq)])
    labels = np.zeros(num_runs * 51)
    valid_trails = 0
    
    onset = np.zeros([num_runs, 51])
    duration = np.zeros([num_runs, 51])
    description = np.zeros([num_runs, 51])

    #Loop to the first 4 runs.
    CheckFiles = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/*.edf')
    if not CheckFiles:
        return 
    
    for runNo in range(num_runs): 
        valid_trails_in_run = 0
        #Get .edf and .json file for following subject and run.
        EDFfile = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.edf')
        JSONfile = glob.glob(data_path + 'S_'+subjectNo[subject].zfill(3) + '/S'+ str(session) +'/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.json')
    
        #Check if EDFfile list is empty
        if not EDFfile:
          continue
    
        # We use mne.read_raw_edf to read in the .edf EEG files
        raw = read_raw_edf(str(EDFfile[0]), preload=True, verbose=False)
        
        # Opening JSON file of the current RUN.
        f = open(JSONfile[0],) 
    
        # returns JSON object as a dictionary 
        JSON = json.load(f) 
    
        #Number of Keystrokes Markers
        keyStrokes = np.min([len(JSON['Markers']), 51]) #len(JSON['Markers']), to avoid extra markers by accident
        # MarkerStart = JSON['Markers'][0]['startDatetime']
           
        #Get Start time of marker
        date_string = EDFfile[0][-21:-4]
        datetime_format = "%d.%m.%y_%H.%M.%S"
        startRecordTime = datetime.strptime(date_string, datetime_format).replace(tzinfo=None)
    
        currentTrialNo = 0 # 1 = fingers, 2 = Wrist, 3 = Elbow, 4 = rest
        if(runNo == 4): 
            currentTrialNo = 4
    
        ch_names = raw.info['ch_names'][4:36]
             
        # filter the data 
        low_cut = preprocessing_dict.get("low_cut", 4.) 
        high_cut = preprocessing_dict.get("high_cut", 50.)
        raw.filter(l_freq=low_cut, h_freq = high_cut)  
        
        raw = raw.copy().pick_channels(ch_names = ch_names)
        # raw = raw.copy().pick(ch_names=ch_names)
        raw.resample(preprocessing_dict.get("sfreq", 128.0))
        fs = raw.info['sfreq']

        for trail in range(keyStrokes):
            
            # class for current trial
            if(runNo == 4 ):               # In Run 5 all trials are 'reset'
                currentTrialNo = 4
            elif (currentTrialNo == 3):    # Set the class of current trial to 1 'Fingers'
                currentTrialNo = 1   
            else:                          # In Runs 1-4, 1st trial is 1 'Fingers', 2nd trial is 2 'Wrist', and 3rd trial is 'Elbow', and repeat ('Fingers', 'Wrist', 'Elbow', ..)
                currentTrialNo = currentTrialNo + 1
                
            trailDuration = 8
            
            trailTime = parse(JSON['Markers'][trail]['startDatetime']).replace(tzinfo=None)
            trailStart = trailTime - startRecordTime
            trailStart = trailStart.seconds 
            start = trailStart + (6 - mi_duration)
            stop = trailStart + 6

            if (trail < keyStrokes-1):
                trailDuration = parse(JSON['Markers'][trail+1]['startDatetime']) - parse(JSON['Markers'][trail]['startDatetime'])
                trailDuration =  trailDuration.seconds + (trailDuration.microseconds/1000000)
                if (trailDuration < 7.5) or (trailDuration > 8.5):
                    print('In Session: {} - Run: {}, Trail no: {} is skipped due to short/long duration of: {:.2f}'.format(session, (runNo+1), (trail+1), trailDuration))
                    if (trailDuration > 14 and trailDuration < 18):
                        if (currentTrialNo == 3):   currentTrialNo = 1   
                        else:                       currentTrialNo = currentTrialNo + 1
                    continue
                
            elif (trail == keyStrokes-1):
                trailDuration = raw[0, int(trailStart*int(fs)):int((trailStart+8)*int(fs))][0].shape[1]/fs
                if (trailDuration < 7.8) :
                    print('In Session: {} - Run: {}, Trail no: {} is skipped due to short/long duration of: {:.2f}'.format(session, (runNo+1), (trail+1), trailDuration))
                    continue

            MITrail = raw[:32, int(start*int(fs)):int(stop*int(fs))][0]
            if (MITrail.shape[1] != data.shape[2]):
                print('Error in Session: {} - Run: {}, Trail no: {} due to the lost of data'.format(session, (runNo+1), (trail+1)))
                return
            
            # select some specific classes
            if ((('Fingers' in classes_labels) and (currentTrialNo==1)) or 
            (('Wrist' in classes_labels) and (currentTrialNo==2)) or 
            (('Elbow' in classes_labels) and (currentTrialNo==3)) or 
            (('Rest' in classes_labels) and (currentTrialNo==4))):
                data[valid_trails] = MITrail
                labels[valid_trails] =  currentTrialNo
                
                # For Annotations
                onset[runNo, valid_trails_in_run]  = start
                duration[runNo, valid_trails_in_run] = trailDuration - (6 - mi_duration)
                description[runNo, valid_trails_in_run] = currentTrialNo
                valid_trails += 1
                valid_trails_in_run += 1
                         
    data = data[0:valid_trails, :, :]
    labels = labels[0:valid_trails]
    labels = (labels-1).astype(int)

    return data, labels, onset, duration, description

def load_REHMI_data(subjects, preprocessing_dict: dict,
                       classes_labels=['Fingers', 'Wrist', 'Elbow', 'Rest'],
                       mi_duration=4):
    """
    Load training/testing data for multiple subjects from the CS2R motor imagery dataset.

    Parameters
    ----------
    subjects : list of int
        List of subject IDs to load data for.
    preprocessing_dict : dict
        Dictionary containing preprocessing parameters.
    classes_labels : list of str, optional
        List of motor imagery class labels to include.
    mi_duration : int, optional
        Duration of the motor imagery segment.

    Returns
    -------
    dict
        Dictionary with 'data' and 'labels' keys, each containing subject-wise dictionaries
        with 'train' and 'test' keys.
    """
    data = {}
    labels = {}

    for subject_id in subjects:
        data_train, labels_train, _, _, _ = load_REHMI(subject_id, training=True,
                                                               preprocessing_dict=preprocessing_dict,
                                                               classes_labels=classes_labels)
        data_test, labels_test, _, _, _ = load_REHMI(subject_id, training=False,
                                                             preprocessing_dict=preprocessing_dict,
                                                             classes_labels=classes_labels)

        data[str(subject_id)] = {
            "train": data_train,
            "test": data_test
        }

        labels[str(subject_id)] = {
            "train": labels_train,
            "test": labels_test
        }

    return {"data": data, "labels": labels}



if __name__ == "__main__":
    preprocessing_config = {
        "low_cut": 4.0,
        "high_cut": 50.0,
        "sfreq": 128.0
    }

    subjects = list(range(3))  # load first 3 subjects
    dataset = load_REHMI_data(subjects, preprocessing_config,
                               classes_labels=['Fingers', 'Wrist', 'Elbow', 'Rest'],
                               mi_duration=4)

    train_data = dataset['data']['0']['train']
    train_labels = dataset['labels']['0']['train']
    print("Shape of Subject 0 Training Data:", train_data.shape)
    print("First 5 Labels:", train_labels[:5])
