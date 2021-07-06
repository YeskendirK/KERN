import yaml
from collections import defaultdict

import numpy as np
import torch
import yaml

from model.KERN import KERN
from model.KERNGeoStyle import KERNGeoStyle
from model.KERN_oms import KERN_oms

torch.manual_seed(1234)
from tqdm import tqdm
import argparse
from utility import select_device


def main(opt):
    conf = yaml.load(open("./config.yaml"))
    dataset_name = opt.dataset  # options: fit, geostyle, omnious
    conf = conf[dataset_name]
    conf["dataset"] = dataset_name

    print("Starting evaluating ...")

    if conf["dataset"] == "fit":
        from utility import TrendData
    elif conf["dataset"] == "omnious":
        from utility_omnious import TrendData
    else:
        from utility_geostyle import TrendData

    dataset = TrendData(conf)
    conf["grp_num"] = len(dataset.grp_id_map)
    conf["ele_num"] = len(dataset.ele_id_map)
    conf["time_num"] = dataset.time_num
    if conf["dataset"] == "fit":
        conf["city_num"] = len(dataset.city_id_map)
        conf["gender_num"] = len(dataset.gender_id_map)
        conf["age_num"] = len(dataset.age_id_map)
    elif conf["dataset"] == "omnious":
        conf["location_num"] = len(dataset.location_id_map)
        conf["segment_num"] = len(dataset.segment_id_map)
        conf["target_age_num"] = len(dataset.target_age_id_map)

    device = select_device(opt.device, batch_size=conf["batch_size"])
    conf["device"] = device
    if conf["dataset"] == "fit":
        model = KERN(conf, dataset.adj)
    elif conf["dataset"] == "omnious":
        model = KERN_oms(conf)
    else:
        model = KERNGeoStyle(conf)
    model.to(device)

    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(opt.weights, map_location=device)
        state_dict = ckpt['model']
        model.load_state_dict(state_dict, strict=False)
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), opt.weights))

    mae_map, mape_map, perGroup_grd, perGroup_pred, all_train, val_mae_map, val_mape_map, test_mae_map, test_mape_map, test_loss_print = \
        evaluate(model, dataset, device, conf)

    grp_names = [key for key in dataset.grp_id_map.keys()]

    print("GROUP, MAE , MAPE , VAL_MAE , VAL_MAPE , TEST_MAE , TEST_MAPE ")
    for grp_name in grp_names:
        print("%s, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (
            grp_name, mae_map[grp_name], mape_map[grp_name], val_mae_map[grp_name], val_mape_map[grp_name],
            test_mae_map[grp_name], test_mape_map[grp_name]))

    print("--" * 10)
    for grp_name in mae_map.keys():
        if grp_name in grp_names:
            continue
        print("%s, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (
            grp_name, mae_map[grp_name], mape_map[grp_name], val_mae_map[grp_name], val_mape_map[grp_name],
            test_mae_map[grp_name], test_mape_map[grp_name]))


def evaluate(model, dataset, device, conf):
    model.eval()
    pbar = tqdm(enumerate(dataset.test_loader), total=len(dataset.test_loader))
    loss_print, enc_loss_print, dec_loss_print = 0, 0, 0
    all_grd, all_pred, all_norm = [], [], []
    perGroup_grd, perGroup_pred, perGroup_norm = defaultdict(list), defaultdict(list), defaultdict(list)

    grp_name_map = {}
    for grp_name, grp_id in dataset.grp_id_map.items():
        grp_name_map[grp_id] = grp_name

    for batch_i, batch in pbar:
        each_cont = [each.to(device) for each in batch]
        enc_loss, dec_loss, pred, _, _, _ = model.predict(each_cont)

        loss = enc_loss + dec_loss
        loss_print += loss.item()
        enc_loss_print += enc_loss.item()
        dec_loss_print += dec_loss.item()

        loss_final = loss_print / (batch_i + 1)
        enc_loss_final = enc_loss_print / (batch_i + 1)
        dec_loss_final = dec_loss_print / (batch_i + 1)

        pbar.set_description(
            'L:{:.6f}, EL:{:.6f}, DL:{:.6f}'.format(loss_print / (batch_i + 1), enc_loss_print / (batch_i + 1),
                                                    dec_loss_print / (batch_i + 1)))
        if conf["dataset"] == "fit":
            [input_seq, output_seq, grp_ids, ele_id, norm, city_id, gender_id, age_id] = each_cont
        elif conf["dataset"] == "omnious":
            [input_seq, output_seq, grp_ids, ele_id, norm, location_id, segment_id, target_age_id] = each_cont
        else:
            [input_seq, output_seq, grp_ids, ele_id, norm] = each_cont

        all_grd.append(output_seq[:, :, 1].cpu())
        all_pred.append(pred.cpu())
        all_norm.append(norm.cpu())
        all_train = dataset.all_train_seq

        for i in range(len(grp_ids)):
            grp_id = grp_ids[i][0].item()
            grp_name = grp_name_map[grp_id]
            perGroup_grd[grp_name].append(torch.stack([output_seq[i, :, 1].cpu()]))
            perGroup_pred[grp_name].append(torch.stack([pred[i].cpu()]))
            perGroup_norm[grp_name].append(torch.stack([norm[i].cpu()]))
            location_name, segment_name, target_age_name = grp_name.split("__")
            for feat_name in [location_name, segment_name, target_age_name]:
                perGroup_grd[feat_name].append(torch.stack([output_seq[i, :, 1].cpu()]))
                perGroup_pred[feat_name].append(torch.stack([pred[i].cpu()]))
                perGroup_norm[feat_name].append(torch.stack([norm[i].cpu()]))

    all_grd = torch.cat(all_grd, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).detach().numpy()
    all_norm = torch.cat(all_norm, dim=0).detach().numpy()
    all_train = dataset.all_train_seq

    for grp_name in perGroup_grd.keys():
        perGroup_grd[grp_name] = torch.cat(perGroup_grd[grp_name], dim=0).numpy()
        perGroup_pred[grp_name] = torch.cat(perGroup_pred[grp_name], dim=0).detach().numpy()
        perGroup_norm[grp_name] = torch.cat(perGroup_norm[grp_name], dim=0).detach().numpy()

    if conf["denorm"] is True:
        all_grd = denorm(all_grd, all_norm)
        all_pred = denorm(all_pred, all_norm)
        all_train = denorm(all_train, all_norm)

        for grp_name in perGroup_grd.keys():
            perGroup_grd[grp_name] = denorm(perGroup_grd[grp_name], perGroup_norm[grp_name])
            perGroup_pred[grp_name] = denorm(perGroup_pred[grp_name], perGroup_norm[grp_name])

    # perGroup_grd['all'] = all_grd
    # perGroup_pred['all'] = all_pred

    val_pred = all_pred[::2]
    val_grd = all_grd[::2]
    val_pred_map, val_grd_map = {}, {}
    # val_pred_map['all'] = val_pred
    # val_grd_map['all'] = val_grd
    for grp_name in perGroup_pred.keys():
        val_pred_map[grp_name] = perGroup_pred[grp_name][::2]
        val_grd_map[grp_name] = perGroup_grd[grp_name][::2]

    test_pred = all_pred[1::2]
    test_grd = all_grd[1::2]
    test_pred_map, test_grd_map = {}, {}
    # test_pred_map['all'] = test_pred
    # test_grd_map['all'] = test_grd
    for grp_name in perGroup_grd.keys():
        test_pred_map[grp_name] = perGroup_pred[grp_name][1::2]
        test_grd_map[grp_name] = perGroup_grd[grp_name][1::2]

    mae = np.mean(np.abs(all_pred - all_grd))
    mape = np.mean(np.abs(all_pred - all_grd) / all_grd) * 100
    mae_map, mape_map = {}, {}
    # mae_map["all"] = mae
    # mape_map['all'] = mape
    for grp_name in perGroup_pred.keys():
        mae_map[grp_name] = np.mean(np.abs(perGroup_pred[grp_name] - perGroup_grd[grp_name]))
        mape_map[grp_name] = np.mean(
            np.abs(perGroup_pred[grp_name] - perGroup_grd[grp_name]) / perGroup_grd[grp_name]) * 100

    val_mae = np.mean(np.abs(val_pred - val_grd))
    val_mape = np.mean(np.abs(val_pred - val_grd) / val_grd) * 100
    val_mae_map, val_mape_map = {}, {}
    # val_mae_map['all'] = val_mae
    # val_mape_map['all'] = val_mape
    for grp_name in perGroup_pred.keys():
        val_mae_map[grp_name] = np.mean(np.abs(val_pred_map[grp_name] - val_grd_map[grp_name]))
        val_mape_map[grp_name] = np.mean(
            np.abs(val_pred_map[grp_name] - val_grd_map[grp_name]) / val_grd_map[grp_name]) * 100

    test_mae = np.mean(np.abs(test_pred - test_grd))
    test_mape = np.mean(np.abs(test_pred - test_grd) / test_grd) * 100
    test_mae_map, test_mape_map = {}, {}
    # test_mae_map['all'] = test_mae
    # test_mape_map['all'] = test_mape
    for grp_name in perGroup_grd.keys():
        test_mae_map[grp_name] = np.mean(np.abs(test_pred_map[grp_name] - test_grd_map[grp_name]))
        test_mape_map[grp_name] = np.mean(
            np.abs(test_pred_map[grp_name] - test_grd_map[grp_name]) / test_grd_map[grp_name]) * 100

    return mae_map, mape_map, perGroup_grd, perGroup_pred, all_train, val_mae_map, val_mape_map, test_mae_map, test_mape_map, [
        loss_final, enc_loss_final, dec_loss_final]


def denorm(seq, norms):
    # seq: [num_samples]
    # norms: [num_samples, 3] 2nd-dim: min, max, eps
    # seq = np.min(seq, 1)
    # seq = np.max(seq, 0)
    seq_len = seq.shape[-1]
    min_v = np.expand_dims(norms[:, 0], axis=1).repeat(seq_len, axis=1)
    max_v = np.expand_dims(norms[:, 1], axis=1).repeat(seq_len, axis=1)
    eps = np.expand_dims(norms[:, 2], axis=1).repeat(seq_len, axis=1)
    denorm_res = seq * (max_v - min_v) + min_v
    return denorm_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fit', help='dataset: fit or geostyle or omnious')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')

    opt = parser.parse_args()
    main(opt)
