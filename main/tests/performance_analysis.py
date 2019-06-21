import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_mean_stat(df, col_name):
    data = df[~df[col_name].isnull()]
    convert_dict = {
                    col_name: float
                    }
    data = data.astype(convert_dict)
    data_slice = data[['num_trackableObjects', col_name]]
    mean_time = data_slice.groupby('num_trackableObjects').mean()

    return mean_time

def get_result(timelog_path):
    print('[TIME LOG] summary...')

    df = pd.read_csv(timelog_path)
    df.columns = ['timestamp', 'log', 'num_trackableObjects',
                  't_detecting', 't_tracking',
                  't_emb_matrix', 't_updating_trObj',
                  't_face_recognition', 't_proc_next_frame']
    print(df['t_emb_matrix'][0])
    df = df.replace(" None", np.nan)

    # mean detecting time for each number of trackableObjects
    detecting = get_mean_stat(df, col_name='t_detecting')

    # mean tracking time for each number of trackableObjects
    tracking = get_mean_stat(df, col_name='t_tracking')

    # mean updating tracking time for each number of trackableObjects
    updating = get_mean_stat(df, col_name='t_updating_trObj')

    # mean face recognition time for each number of trackableObjects
    face_recognition = get_mean_stat(df, col_name='t_face_recognition')

    frames = [detecting, tracking, updating, face_recognition]
    report = pd.concat(frames, axis=1)
    print(report)

    report.to_csv('tests/logs/report_{}.csv'.format(timelog_path.split('/')[-1].split('.')[0]))

    return report