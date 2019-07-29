import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_mean_stat(df, col_name):
    data = df[~df[col_name].isnull()]
    convert_dict = {col_name: float}
    data = data.astype(convert_dict)
    data_slice = data[['num_trackableObjects', col_name]]
    mean_time = data_slice.groupby('num_trackableObjects').mean()
    # mean_time = mean_time[mean_time.index != 'None']

    fig, axes = plt.subplots(figsize=(12, 6))
    axes.set_xlabel('num_trObj')
    axes.set_ylabel('avg_time (sec)')
    mean_time.plot(ax=axes);

    return mean_time, fig

def get_result(vs, timelog_path, save_dir, face_detection):
    df = pd.read_csv(timelog_path)
    df.columns = ['timestamp', 'log', 'num_trackableObjects',
                  't_detecting', 't_tracking',
                  't_emb_matrix', 't_updating_trObj',
                  't_face_recognition', 't_proc_next_frame']

    df = df.replace(" None", np.nan)

    frames = []
    # mean stage processing time for each number of trackableObjects
    if face_detection == 'True':
        to_analyze = ['t_detecting', 't_tracking', 't_updating_trObj', 't_face_recognition']
    else:
        to_analyze = ['t_detecting', 't_tracking', 't_updating_trObj']

    for col_name in to_analyze:
        stage_frame, fig = get_mean_stat(df, col_name)
        fig.savefig(os.path.join(save_dir, 'avg_time_{}.jpg'.format(col_name)))
        frames.append(stage_frame)

    report = pd.concat(frames, axis=1, sort=True)
    report_file = os.path.join(save_dir, 'report_{}.csv'.format(timelog_path.split('/')[-1].split('.')[0]))
    report.to_csv(report_file)
    print(report)

    vs.test_info.append('[PERFOMANCE RESULT INFO] saved to {}\n'.format(report_file))