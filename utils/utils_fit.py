from nets.unet_training import CE_Loss, Dice_loss
from tqdm import tqdm
import torch.nn.functional as F
import os

import torch
import numpy as np
import torch.nn as nn
from utils.utils import get_lr
from utils.utils_metrics import f_score
from utils.dataloader import augmentationimage as ugmentationimage
from utils.TI_loss import TI_Loss


# --------------------------------------------------------------------------- #
# NOTE (added while patching this file, not part of the original release):
#
# As shipped, LTSeg's train.py sets loss_fuc = "TPCloss", but this file's
# if/elif chain only handles "BCEloss" / "Diceloss" / "LTSloss" -- so with the
# repo's own defaults, `loss` is never assigned and training crashes with
# NameError on the very first iteration. There is also a call to
# `trainDice_loss(...)` inside the "LTSloss" branch that is never defined or
# imported anywhere in the repository.
#
# The paper's actual "connectivity + interactivity prior" loss (matching the
# critical-pixel map computed just above the trainDice_loss call) was not
# published. `trainDice_loss` below is MY reconstruction of a plausible
# masked/weighted Dice loss that uses that critical-pixel map the same way
# the TI-Loss family of papers does (see NexToU eq. 5: L = L_pixel(f⊙V, g⊙V)).
# It is NOT verified against the authors' results and should be treated as a
# placeholder to get the pipeline running end-to-end, not a faithful
# reproduction of the MICCAI paper's numbers. Swap it out if you get the
# authors' real implementation.
# --------------------------------------------------------------------------- #
def trainDice_loss(inputs, target, criticals_map, beta=1, smooth=1e-5):
    """
    Dice loss computed only over the "critical" pixels flagged by the
    connectivity/interactivity map (criticals_map), reconstructed loss --
    see note above.
    """
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    probs = torch.sigmoid(inputs)                       # [n, c, h, w]
    probs = probs.permute(0, 2, 3, 1).contiguous()       # [n, h, w, c]
    gt = target[..., :-1]                                # drop the "ignore/background" channel, matches Dice_loss

    # criticals_map is [n, c, h, w] in utils_fit.py -> align to [n, h, w, c]
    crit = criticals_map.permute(0, 2, 3, 1).contiguous()
    crit = crit[..., :c]

    probs = probs * crit
    gt = gt * crit

    tp = torch.sum(gt * probs, dim=[0, 1, 2])
    fp = torch.sum(probs, dim=[0, 1, 2]) - tp
    fn = torch.sum(gt, dim=[0, 1, 2]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    return 1 - torch.mean(score)


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, loss_fuc, num_classes, save_dir, no_improve_count):
    total_loss = 0
    val_loss = 0
    val_f_score = 0


    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, ascii=True)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs = batch
        with torch.no_grad():
            imgs = imgs.cuda()  # [bsz, 3, 448, 448]
            pngs = pngs.cuda()  # tragets

        optimizer.zero_grad()
        outputs = model_train(imgs)  # [bsz, 24, 448, 448]

        # ----------------------------------
        # choose the loss fuc
        # ----------------------------------
        if loss_fuc == "BCEloss":
            loss = CE_Loss(outputs, pngs)

        elif loss_fuc == "Diceloss":
            loss = Dice_loss(outputs, pngs)

        elif loss_fuc in ("LTSloss", "TPCloss"):
            # "TPCloss" is what train.py actually sets by default; it is
            # routed to the same branch as "LTSloss" here since no separate
            # "TPCloss" implementation exists in the released code.

            sigoutputs = torch.sigmoid(outputs.clone())
            pred_binary = sigoutputs * 255
            pred_binary[pred_binary <= 127] = 0
            pred_binary[pred_binary > 127] = 1
            ####################################
            repred_binary = sigoutputs * 255
            repred_binary[repred_binary <= 127] = 1
            repred_binary[repred_binary > 127] = 0
            ####################################
            true_binary = pngs.clone()
            interlist = []
            contlist = []

            np_kernel = torch.tensor([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]], dtype=torch.float32)
            connectivity_kernel = torch.unsqueeze(torch.unsqueeze(np_kernel, 0), 0).cuda()

            for i in range(20):
                copied_pred_overlap = pred_binary[:, i:i + 1, :, :].expand(-1, 20, -1, -1)
                copied_true_binary = true_binary[:, i:i + 1, :, :].expand(-1, 20, -1, -1)
                pred_overlap = copied_pred_overlap * true_binary
                true_overlap = copied_true_binary * true_binary
                rev_true_overlap = torch.logical_not(true_overlap)
                true_pred_overlap = pred_overlap * rev_true_overlap
                true_pred_overlap = torch.sum(true_pred_overlap, dim=1, keepdim=True)

                interlist.append(true_pred_overlap)

                pred_channel = pred_binary[:, i, :, :].unsqueeze(1)

                expend_pred = F.conv2d(pred_channel, connectivity_kernel, stride=1, padding=2)   # 5*5

                expend_pred = expend_pred.squeeze(1)
                expend_pred[expend_pred <= 1] = 0
                expend_pred[expend_pred > 1] = 1

                truepred = expend_pred * repred_binary[:, i, :, :]

                intersection_map = truepred * true_binary[:, i, :, :]

                contlist.append(intersection_map)

            criticals_conet_map = torch.stack(contlist, dim=1)
            criticals_inter_map = torch.cat(interlist, dim=1)
            criticals_map = criticals_inter_map + criticals_conet_map
            criticals_map[criticals_map <= 0] = 0
            criticals_map[criticals_map > 0] = 1

            loss = trainDice_loss(outputs, pngs, criticals_map)
        else:
            raise ValueError(f"Unknown loss_fuc '{loss_fuc}'. Expected one of: BCEloss, Diceloss, LTSloss, TPCloss")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()

    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, ascii=True)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        imgs, pngs = batch

        with torch.no_grad():
            imgs = imgs.cuda()  # [bsz, 3, 448, 448]
            pngs = pngs.cuda()  # tragets

            outputs = model_train(imgs)
            # ----------------------------------
            # choose the loss fuc
            # ----------------------------------
            if loss_fuc == "BCEloss":
                loss = CE_Loss(outputs, pngs)
            elif loss_fuc == "Diceloss":
                loss = Dice_loss(outputs, pngs)
            elif loss_fuc in ("LTSloss", "TPCloss"):
                loss = Dice_loss(outputs, pngs)
            val_loss += loss.item()


        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                            'f_score': val_f_score / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    if epoch > 100 and epoch % 50 == 0:

        save_path = os.path.join(save_dir, f"epoch_{epoch}_weights.pth")
        torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    no_improve_count += 1
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        no_improve_count = 0

    return no_improve_count
