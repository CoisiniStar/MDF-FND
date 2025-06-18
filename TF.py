import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

from utils import config


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


def train_func_epoch(epoch, model, data_loader, device, optimizer, scheduler, lambda_i, lambda_t):
    # Put the model into the training mode
    model.train()
    total_loss = 0
    targets = []
    predictions = []
    index = 0
    ## Start a tqdm bar
    with tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch:
        for step, batch in enumerate(single_epoch):
            ## To store values for each batch in an epoch

            single_epoch.set_description(f"Training- Epoch {epoch}")
            # print(batch.__len__())
            # batched_img, batched_txt, batch_labels = batch
            batched_img, batched_txt, batch_labels = zip(*batch)
            # print(batched_img)

            ## To tensor and Load the inputs to the device
            ##
            img_list = list(batched_img)
            txt_list = list(batched_txt)
            label_list = list(batch_labels)
            # label_stacked = torch.stack(label_list)
            img_stacked = torch.stack(img_list)
            txt_stacked = torch.stack(txt_list)
            img_stacked = img_stacked.to(device)
            txt_stacked = txt_stacked.to(device)

            label_tensor = torch.tensor(label_list)
            batch_labels = label_tensor.to(device)

            # Perform a forward pass. This will return Multimodal vec and total loss.
            batch_logits,margin_loss = model(img_stacked, txt_stacked)

            # list to tensor
            # batch_logits = torch.tensor(batch_logits)
            stack_batch = torch.stack(batch_logits)
            stack_batch = stack_batch.squeeze(1)
            pred_multimodal = torch.argmax(stack_batch, dim=1).flatten().cpu().numpy()
            predictions.append(pred_multimodal)
            targets.append(batch_labels.cpu().numpy())
            # predictions.append(batch_logits)
            # targets.append(batch_labels.cpu().numpy())
            # predictions[index] = torch.FloatTensor(predictions[index]).squeeze()
            # print(predictions.shape)
            # targets[index] = torch.FloatTensor(targets[index]).squeeze()


            # print(targets.shape)
            ## Calculate the final loss
            loss = F.cross_entropy(stack_batch, batch_labels) + config.margin_weight * margin_loss
            # loss = F.cross_entropy(stack_batch, batch_labels)+margin_loss
            # index += 1
            # loss = F.cross_entropy(predictions, batch_labels) + lambda_i * kl_img + lambda_i * kl_txt # 计算总损失
            # loss = loss.requires_grad_()

            # total_loss += loss.item()
            total_loss += loss.item()
            loss.backward()

            if step % config.gradient_accumulation_steps == 0 or step == len(data_loader) - 1:
                ## Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                ## torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Zero out any previously calculated gradients
                model.zero_grad()

            ## Update tqdm bar
            single_epoch.set_postfix(train_loss=total_loss / (step + 1))

    ## Create single vector for predictions and ground truth

    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    ## Calculate performance metrics
    report = classification_report(targets, predictions, output_dict=True, labels=[0, 1])

    ## Average out the loss
    epoch_train_loss = total_loss / len(data_loader)

    return epoch_train_loss, report
