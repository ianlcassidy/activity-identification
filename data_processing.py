#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob

###############################################################################

# column data from the readme.pdf file
col_names = ['timestamp', 'activityID', 'heartrate',
             'hand_temp', 'hand_accel_x', 'hand_accel_y', 'hand_accel_z',
             'hand_bad1', 'hand_bad2', 'hand_bad3',
             'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
             'hand_magnet_x', 'hand_magnet_y', 'hand_magnet_z',
             'hand_orient_1', 'hand_orient_2', 'hand_orient_3', 'hand_orient_4',
             'chest_temp', 'chest_accel_x', 'chest_accel_y', 'chest_accel_z',
             'chest_bad1', 'chest_bad2', 'chest_bad3',
             'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
             'chest_magnet_x', 'chest_magnet_y', 'chest_magnet_z',
             'chest_orient_1', 'chest_orient_2', 'chest_orient_3', 'chest_orient_4',
             'ankle_temp', 'ankle_accel_x', 'ankle_accel_y', 'ankle_accel_z',
             'ankle_bad1', 'ankle_bad2', 'ankle_bad3',
             'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
             'ankle_magnet_x', 'ankle_magnet_y', 'ankle_magnet_z',
             'ankle_orient_1', 'ankle_orient_2', 'ankle_orient_3', 'ankle_orient_4']

# subject info data from the subjectInformation.pdf file
subject_info = {'SubjectID': [101, 102, 103, 104, 105, 106, 107, 108, 109],
                'Sex': ['Male', 'Female', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male'],
                'Age': [27, 25, 31, 24, 26, 26, 23, 32, 31],
                'Height': [182, 169, 187, 194, 180, 183, 173, 179, 168],
                'Weight': [83, 78, 92, 95, 73, 69, 86, 87, 65],
                'RestingHR': [75, 74, 68, 58, 70, 60, 60, 66, 54],
                'MaxHR': [193, 195, 189, 196, 194, 194, 197, 188, 189],
                'DominantHand': ['right', 'right', 'right', 'right', 'right', 'right', 'right', 'left', 'right']}
subject_info = pd.DataFrame(subject_info)
subject_info = subject_info.set_index('SubjectID')


def create_features_and_labels(df, subjectID):
    activity_array = np.unique(df['activityID'])
    # remove activityID 0 because it is a "transient activity"
    activity_array = activity_array[activity_array != 0]

    X = []
    Y = []

    for act in activity_array:
        df_ID = df[df['activityID'] == act]

        # find 0.1 and 0.9 indices
        i1 = len(df_ID) / 10
        i2 = 9 * len(df_ID) / 10

        # initialize features
        X_obs = np.array([])
        # mean temperature (all three locations)
        X_obs = np.append(X_obs, (np.nanmean(df_ID['hand_temp'][i1:i2]),
                                  np.nanmean(df_ID['chest_temp'][i1:i2]),
                                  np.nanmean(df_ID['ankle_temp'][i1:i2])))
        # mean accel (all three locations)
        X_obs = np.append(X_obs, (np.nanmean(df_ID['hand_accel_x'][i1:i2]),
                                  np.nanmean(df_ID['hand_accel_y'][i1:i2]),
                                  np.nanmean(df_ID['hand_accel_z'][i1:i2]),
                                  np.nanmean(df_ID['chest_accel_x'][i1:i2]),
                                  np.nanmean(df_ID['chest_accel_y'][i1:i2]),
                                  np.nanmean(df_ID['chest_accel_z'][i1:i2]),
                                  np.nanmean(df_ID['ankle_accel_x'][i1:i2]),
                                  np.nanmean(df_ID['ankle_accel_y'][i1:i2]),
                                  np.nanmean(df_ID['ankle_accel_z'][i1:i2])))
        # mean abs gyro (all three locations)
        X_obs = np.append(X_obs, (np.nanmean(np.abs(df_ID['hand_gyro_x'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['hand_gyro_y'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['hand_gyro_z'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['chest_gyro_x'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['chest_gyro_y'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['chest_gyro_z'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['ankle_gyro_x'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['ankle_gyro_y'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['ankle_gyro_z'][i1:i2]))))
        # mean abs magnet (all three locations)
        X_obs = np.append(X_obs, (np.nanmean(np.abs(df_ID['hand_magnet_x'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['hand_magnet_y'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['hand_magnet_z'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['chest_magnet_x'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['chest_magnet_y'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['chest_magnet_z'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['ankle_magnet_x'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['ankle_magnet_y'][i1:i2])),
                                  np.nanmean(np.abs(df_ID['ankle_magnet_z'][i1:i2]))))
        # heartrate difference (beginning, middle, and end of activity)
        i1 = 33 * len(df_ID) / 100
        i2 = 66 * len(df_ID) / 100
        i3 = len(df_ID) - 1
        X_obs = np.append(X_obs, (np.nanmean(df_ID['heartrate'][0:i1]) - subject_info['RestingHR'][subjectID],
                                  np.nanmean(df_ID['heartrate'][i1:i2]) - subject_info['RestingHR'][subjectID],
                                  np.nanmean(df_ID['heartrate'][i2:i3]) - subject_info['RestingHR'][subjectID]))
        # subject info 
        X_obs = np.append(X_obs, (subject_info['Age'][subjectID],
                                  subject_info['Height'][subjectID],
                                  subject_info['Weight'][subjectID]))
        # sex (one-hot-encode)
        if subject_info['Sex'][subjectID] == 'Male':
            X_obs = np.append(X_obs, (0, 1))
        else:
            X_obs = np.append(X_obs, (1, 0))
        # dominant hand (one-hot-encode)
        if subject_info['DominantHand'][subjectID] == 'Right':
            X_obs = np.append(X_obs, (0, 1))
        else:
            X_obs = np.append(X_obs, (1, 0))
        # remove any NaN values
        X_obs = np.nan_to_num(X_obs)

        # append features and labels
        X.append(X_obs)
        Y.append(act)

    return np.vstack(X), np.array(Y)

###############################################################################


def main():
    file_loc_array = glob.glob('PAMAP2_Dataset/protocol/*.dat')
    file_loc_array = file_loc_array + \
        glob.glob('PAMAP2_Dataset/optional/*.dat')

    if len(file_loc_array) == 0:
        print 'PAMAP2 dataset folder not found.'
        print 'Please download the dataset from:'
        print 'http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring'
        print 'and copy to local directory.'
        raise Exception

    X = []
    Y = []
    for file_loc in file_loc_array:
        print 'processing file: %s' % file_loc

        subjectID = int(file_loc[31:34])
        df = pd.read_csv(file_loc, sep=' ', names=col_names)

        Xf, Yf = create_features_and_labels(df, subjectID)
        X.append(Xf)
        Y.append(Yf)

    X = np.vstack(X)
    Y = np.concatenate(Y)

    print 'Saving X and Y to CSV files.'
    np.savetxt('X_train.csv', X, delimiter=',')
    np.savetxt('Y_train.csv', Y, delimiter=',')

if __name__ == '__main__':
    main()
