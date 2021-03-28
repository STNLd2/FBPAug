import argparse
import os
from pathlib import Path

import numpy as np
from resource_manager import read_config
from tqdm import tqdm
import pandas as pd

from dpipe.io import save_json, load_numpy
from dpipe.im.metrics import dice_score
from sinogram_augmentation.metrics import surface_dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', required=True, type=str)
    parser.add_argument('--n_exp', required=True, type=str)

    args = parser.parse_known_args()[0]
    exp_path = Path(args.exp_path)
    n = args.n_exp

    cfg = read_config(exp_path / 'resources.config')
    print(f'\n>>> Metric for `experiment_{n}`\n', flush=True)
    df = pd.read_csv(f'{os.path.realpath("..")}/cancer-500-paired-kernels.csv')
    test_ids1 = list(df['SeriesInstanceUID_1'].values)
    test_ids2 = list(df['SeriesInstanceUID_2'].values)

    dice_records, sdice05_records, sdice10_records, sdice20_records = {}, {}, {}, {}
    for id1, id2 in tqdm(zip(test_ids1, test_ids2)):
            predict1 = load_numpy(exp_path / f'experiment_{n}/test_predictions_cancer500/{id1}.npy', decompress=True)
            predict2 = load_numpy(exp_path / f'experiment_{n}/test_predictions_cancer500/{id2}.npy', decompress=True)
            if predict1.shape == predict2.shape:
                spacing = cfg.dataset_to_test.spacing(id1)
                if np.sum(predict1 >= 0.5) == 0 or np.sum(predict2 >= 0.5) == 0:
                    dice_records[f'{id1}_{id2}'] = 0.0
                    sdice05_records[f'{id1}_{id2}'] = 0.0
                    sdice10_records[f'{id1}_{id2}'] = 0.0
                else:
                    dice_records[f'{id1}_{id2}'] = dice_score(predict1 >= 0.5, predict2 >= 0.5)
                    sdice10_records[f'{id1}_{id2}'] = surface_dice(predict1 >= 0.5, predict2 >= 0.5,
                                                                   spacing=spacing, tolerance=1.0)
                    sdice05_records[f'{id1}_{id2}'] = surface_dice(predict1 >= 0.5, predict2 >= 0.5,
                                                                   spacing=spacing, tolerance=0.5)


    dice_path = exp_path / f'experiment_{n}/test_pairwise_metrics/dice.json'
    sdice05_path = exp_path / f'experiment_{n}/test_pairwise_metrics/sdice_0.5.json'

    save_json(dice_records, dice_path, indent=0)
    save_json(sdice05_records, sdice05_path, indent=0)
