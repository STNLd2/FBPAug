import argparse
from pathlib import Path

from resource_manager import read_config
from tqdm import tqdm
from dpipe.io import save_json, load_numpy
from dpipe.im import zoom_to_shape
from dpipe.im.metrics import dice_score
from fbp_aug.metrics import surface_dice
from fbp_aug.utils import get_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', required=True, type=str)
    parser.add_argument('--n_exp', required=True, type=str)

    args = parser.parse_known_args()[0]
    exp_path = Path(args.exp_path)
    n = args.n_exp

    cfg = read_config(exp_path / 'resources.config')
    print(f'\n>>> Metric for `experiment_{n}`\n', flush=True)
    uids = get_ids(f'{os.path.realpath("..")}/cancer-500-paired-kernels.csv')
    dataset_to_test = cfg.dataset_to_test

    dice_records, sdice05_records, sdice10_records, sdice20_records = {}, {}, {}, {}
    for uid in tqdm(uids):
        predict = load_numpy(exp_path / f'experiment_{n}/test_predictions_cancer500/{uid}.npy', decompress=True)
        target = dataset_to_test.lungs(uid)
        target = zoom_to_shape(target, predict.shape)
        spacing = cfg.dataset_to_test.spacing(uid)
        dice_records[uid] = dice_score(predict >= 0.5, target >= 0.5)
        sdice10_records[uid] = surface_dice(predict >= 0.5, target >= 0.5,
                                            spacing=spacing, tolerance=1.0)
        sdice05_records[uid] = surface_dice(predict >= 0.5, target >= 0.5,
                                            spacing=spacing, tolerance=0.5)

    dice_path = exp_path / f'experiment_{n}/test_cancer500_metrics/dice.json'
    sdice10_path = exp_path / f'experiment_{n}/test_cancer500_metrics/sdice_1.0.json'

    save_json(dice_records, dice_path, indent=0)
    save_json(sdice10_records, sdice10_path, indent=0)
