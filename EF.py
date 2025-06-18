import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

from utils import config
# import bert_model

def model_metric(tn, fp, fn, tp):
    """Calculate Accuracy, precision, recall and F1-score
    计算准确性，精密度，召回率和f1得分
    Args:
        tn (float)
        fn (float)
        fp (float)
        tp (float)
    """

    acc = ((tp + tn) / (tp + fp + tn + fn)) * 100
    if tp == 0:
        prec = 0
        rec = 0
        f1_score = 0
    else:
        ## calculate the Precision
        prec = (tp / (tp + fp)) * 100

        ## calculate the Recall
        rec = (tp / (tp + fn)) * 100

        ## calculate the F1-score
        f1_score = 2 * prec * rec / (prec + rec)

    return acc, prec, rec, f1_score


def eval_func(model, data_loader, device, criterion,epoch=1):
    """Function for a single validation epoch

    Args:
        epoch (int): current epoch number
        model (nn.Module)
        data_loader (DataLoader)
        device (str)
    """

    # Put the model into the training mode
    model.eval()

    total_loss = 0

    ## To store values for each batch in an epoch
    targets = []
    predictions = []
    # index = 0
    ## Start a tqdm bar
    with tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch:
        for step, batch in enumerate(single_epoch):
            single_epoch.set_description(f"Evaluating- Epoch {epoch}")

            batched_img, batched_txt, batch_labels = zip(*batch)
            img_list = list(batched_img)
            txt_list = list(batched_txt)
            label_list = list(batch_labels)
            # label_stacked = torch.stack(label_list)
            img_stacked = torch.stack(img_list)
            txt_stacked = torch.stack(txt_list)
            ## Load the inputs to the device

            img_stacked = img_stacked.to(device)
            txt_stacked = txt_stacked.to(device)
            label_tensor = torch.tensor(label_list)
            batch_labels = label_tensor.to(device)

            with torch.no_grad():
                batch_logits,margin_loss = model(img_stacked, txt_stacked)

            # predictions.append(batch_logits)
            # targets.append(batch_labels.cpu().numpy())
            # predictions[index] = torch.FloatTensor(predictions[index]).squeeze()
            # targets[index] = torch.FloatTensor(targets[index]).squeeze()
            # loss = criterion(batch_logits, batch_labels)
            # print(targets.shape)
            ## Calculate the final loss
            stack_batch = torch.stack(batch_logits)
            stack_batch = stack_batch.squeeze(1)
            # batch_logits = torch.tensor(stack_batch)
            batch_logits = stack_batch.clone().detach().requires_grad_(True)
            # loss =  F.cross_entropy(batch_logits, batch_labels)
            loss =  F.cross_entropy(batch_logits, batch_labels) + config.margin_weight * margin_loss


            # index += 1
            # loss = F.cross_entropy(predictions, batch_labels) + lambda_i * kl_img + lambda_i * kl_txt # 计算总损失
            # loss = loss.requires_grad_()

            # total_loss += loss.item()

            # prem
            total_loss += loss.item()

            ## Update the tqdm bar
            single_epoch.set_postfix(loss=loss.item())

            # Finding predictions
            pred_multimodal = torch.argmax(batch_logits, dim=1).flatten().cpu().numpy()

            predictions.append(pred_multimodal)
            targets.append(batch_labels.cpu().numpy())
    ## Create single vector for predictions and ground truth
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    ## Avg out the loss
    epoch_validation_loss = total_loss / len(data_loader)

    ## Find the performance metrics
    report = classification_report(targets, predictions, output_dict=True, labels=[0, 1])

    ## Find the confusion metrics
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()

    ## Calculate Micro - metrics from own function
    acc, prec, rec, f1_score = model_metric(tn, fp, fn, tp)

    return epoch_validation_loss, report, acc, prec, rec, f1_score