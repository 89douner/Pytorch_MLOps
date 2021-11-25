import time
import copy
import torch
import numpy as np
import gc
import os 
from utils import EarlyStopping, save_net

def train_model(dataloaders, dataset_sizes, num_iteration, net, criterion, optim, scheduler, device, w_config, classes_name, wandb, patience, ckpt_dir):
    wandb.watch(net, criterion, log='all', log_freq=10)

    since = time.time()
    num_epoch = w_config.epochs
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss, best_acc = 100, 0
    
    classes_name = classes_name
    #label_name = [i for i in range(len(classes_name))]

    early_stopping = EarlyStopping(patience = patience, verbose = True)

    for epoch in range(1, num_epoch+1):
        all_labels, all_preds, all_prob = [], [], []

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            loss_arr = []
            running_corrects = 0
            running_loss = 0

            #train dataset 로드하기
            for iteration_th, (inputs, labels) in enumerate(dataloaders[phase]): #iteration_th: 몇 번재 iteration 인지 알려 줌 "ex) batch_th=0 ← 첫 번째 batch 시작"
                
                ###########################GPU에 데이터 업로드##########################
                inputs = inputs.to(device) #image 데이터 GPU에 업로드
                labels = labels.to(device) #label 데이터 GPU에 업로드 // {labels: 0 → Normal, labels: 1 → Pneumonia} <== alb_data_load_classification.py 참고
                ########################################################################

                # backward pass ← zero the parameter gradients
                optim.zero_grad()

                with torch.set_grad_enabled(phase == "train"): # track history if only in train
                    outputs = net(inputs) #output 결과값은 softmax 입력 직전의 logit 값들
                    _, preds = torch.max(outputs, 1) #pred: 0 → Normal <== labels 참고
                    #preds2 = outputs.sigmoid() > 0.5
                    loss = criterion(outputs, labels) #criterion에 output이 들어가면 softmax 이 후의 확률 값으로 변하고, 변환된 확률 값과 label을 비교하여 loss 계산

                    loss_arr += [loss.item()] #Iteration 당 Loss 계산


                    if phase == "train":
                        loss.backward() #계산된 loss에 의해 backward (gradient) 계산
                        optim.step() #계산된 gradient를 참고하여 backpropagation으로 update
                        
                        wandb.log({"Train Iteration loss": np.mean(loss_arr), 'Iteration_step': iteration_th}, commit=False)
                        print("TRAIN: EPOCH %04d / %04d | ITERATION %04d / %04d | LOSS %.4f" %
                        (epoch, num_epoch, iteration_th, num_iteration['train'], np.mean(loss_arr)))
                        
                    elif phase == 'val':
                        all_labels.extend(labels.detach().cpu().numpy())
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_prob.extend(outputs.detach().cpu().numpy())

                        print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                                (epoch, num_epoch, iteration_th, num_iteration['val'], np.mean(loss_arr))) 

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and w_config.warm_up == 'yes':
                scheduler.step_ReduceLROnPlateau(np.mean(loss_arr)) #←← warm-up 사용 시 learning rate scheduler 실행
            elif phase == 'train' and w_config.warm_up == 'no':
                scheduler.step(np.mean(loss_arr)) #←← warm-up 사용하지 않을 시 learning rate scheduler 실행

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                wandb.log({'train_epoch_loss': epoch_loss, 'train_epoch_acc': epoch_acc}, commit=False)
            
            elif phase == 'val':
                wandb.log({'val_epoch_loss': epoch_loss, 'val_epoch_acc': epoch_acc}, step=epoch)   

            print('Epoch {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            save_net(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch, is_best=False)

            if phase=='val':
                if epoch_loss < best_loss:
                    best_all_labels, best_all_preds, best_all_prob = [], [], []
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(net.state_dict())
                    
                    best_all_labels = all_labels
                    best_all_preds = all_preds
                    best_all_prob = all_prob

                if epoch_acc > best_acc :
                    best_acc = epoch_acc
                    
                wandb.log({'best_acc': best_acc}, commit=False)
                wandb.log({'best_loss': best_loss}, commit=False)

                print('Epoch {} Best Loss: {:.4f} Best Acc: {:.4f}'.format(phase, best_loss, best_acc))

        if phase=='val':
            early_stopping(epoch_loss, net)

            if early_stopping.early_stop:
                wandb_log(wandb, best_all_labels, best_all_preds, best_all_prob, classes_name)
                print("Early stopping")
                break

            elif epoch == num_epoch:
                wandb_log(wandb, best_all_labels, best_all_preds, best_all_prob, classes_name)

        gc.collect()
        torch.cuda.empty_cache()
        print()
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    
    save_net(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch, is_best=True, best_acc=best_acc)
    return net

def wandb_log(wandb, best_all_labels, best_all_preds, best_all_prob, classes_name):
    # ROC, Precision Recall, Confusion Matrix penel 생성
    wandb.log({'ROC curve': wandb.plots.ROC(best_all_labels, best_all_prob, classes_name)}, commit=False)
    wandb.log({'Precision Recall': wandb.plots.precision_recall(best_all_labels, best_all_prob, classes_name)}, commit=False)
    wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix(preds=best_all_preds, y_true=best_all_labels, class_names=classes_name)}, commit=False)