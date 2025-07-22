import os.path

import time
import torch
import gol
from dataset import getDatasets, collate_edge
from torch.utils.data import DataLoader
from model import LaMDA
from evaluation import eval_model
from pprint import pformat

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

def train_eval(model: LaMDA, _all_ds):
    trn_set, val_set, tst_set = _all_ds
    trn_loader = DataLoader(trn_set, batch_size=gol.BATCH_SZ, shuffle=True, collate_fn=collate_edge)
    opt = torch.optim.AdamW(model.parameters(), lr=gol.conf['lr'], weight_decay=gol.conf['decay'])
    batch_num = len(trn_set) // gol.BATCH_SZ
    best_val_epoch, best_val_ndcg, best_val_recall = 0, 0., 0.
    ave_tot, ave_rec, ave_fis = 0., 0., 0.
    tst_result = None

    for epoch in range(gol.EPOCH):
        epoch_train_start_time = time.time()
        model.train()
        for idx, batch in enumerate(trn_loader):

            rec_loss= model.getTrainLoss(batch)
            tot_loss = rec_loss

            opt.zero_grad()
            tot_loss.backward()
            opt.step()
            if idx % (batch_num // 5) == 0:
                gol.pLog(f'Batch {idx} / {batch_num}, Loss: {tot_loss.item():.5f}')

            ave_tot += tot_loss.item()
            ave_rec += rec_loss.item()
        epoch_train_duration = time.time() - epoch_train_start_time

        ave_tot /= batch_num
        ave_rec /= batch_num
        val_results, _ = eval_model(model, val_set)

        gol.pLog(f'Epoch {epoch} / {gol.EPOCH}, Train Time: {epoch_train_duration:.2f}, Loss: {ave_tot:.5f}')
        gol.pLog(f'Valid NDCG@5: {val_results["ndcg"][2]:.5f}, Recall@2: {val_results["recall"][1]:.5f}, Recall@5: {val_results["recall"][2]:.5f}')
        if epoch - best_val_epoch == gol.patience:
            gol.pLog(f'Stop training after {gol.patience} epochs without valid improvement.')
            break

        if val_results["recall"][2] > best_val_recall or epoch == 0:
            best_val_epoch, best_val_ndcg, best_val_recall = epoch, val_results["ndcg"][2], val_results["recall"][2]

            test_inference_start_time = time.time()
            tst_result, _ = eval_model(model, tst_set)

            test_inference_duration = time.time() - test_inference_start_time
            gol.pLog(f'Test Inference Time: {test_inference_duration:.2f}s')
            gol.pLog(f'New test result:\n {pformat(tst_result)}')
            if gol.SAVE:
                torch.save(model.cpu().state_dict(), w_path)
                model.to(gol.device)

        gol.pLog(f'Best valid Recall@5 at epoch {best_val_epoch}')
        gol.pLog(f'Test NDCG@5: {tst_result["ndcg"][2]:.5f}, Recall@2: {tst_result["recall"][1]:.5f}, Recall@5: {tst_result["recall"][2]:.5f}\n')

    return tst_result, best_val_epoch

if __name__ == '__main__':
    w_path = os.path.join(gol.FILE_PATH, 'weight.pth')
    n_user, n_poi, all_ds, geo_graph = getDatasets(gol.DATA_PATH, gol.dataset.lower())
    recModel = LaMDA(n_user, n_poi, geo_graph)
    if gol.LOAD:
        recModel.load_state_dict(torch.load(w_path))
    recModel = recModel.to(gol.device)
    gol.pLog(f'Start Training')
    gol.pLog(f'Dropout: {1 - gol.conf["keepprob"] if gol.conf["dropout"] else 0}\n')

    test_result, best_epoch = train_eval(recModel, all_ds)
    gol.pLog(f'Training on {gol.dataset.upper()} finished, best valid Recall@20 at epoch {best_epoch}')
    gol.pLog(f'Best result:\n{pformat(test_result)}')