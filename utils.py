import torch
import numpy as np
import os
from config import CKPT_DIR
import datetime

def save_net(ckpt_dir, net, optim, epoch, is_best = False, best_acc=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    now = datetime.date.today()
    if is_best == False:
        torch.save(
            {'net': net.state_dict(),'optim': optim.state_dict()},
            os.path.join(ckpt_dir, f'model_epoch_{epoch:04}_{now}.pth'),
        )
    elif is_best == True:
        torch.save(
            {'net': net.state_dict(),'optim': optim.state_dict()},
            os.path.join(ckpt_dir, f'best_model_acc_{best_acc}_{now}.pth'),
        )
        
def load_net(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda fname: int(''.join(filter(str.isdigit, fname))))
    
    ckpt_path = os.path.join(CKPT_DIR, ckpt_list[-1])
    model_dict = torch.load(ckpt_path)
    print(f'* Load {ckpt_path}')

    net.load_state_dict(model_dict['net'])
    optim.load_state_dict(model_dict['optim'])
    epoch = int(''.join(filter(str.isdigit, ckpt_list[-1])))
    
    return net, optim, epoch

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, net):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, net)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, net)
            self.counter = 0

    def save_checkpoint(self, val_loss, net):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(net.state_dict(), self.path)
        self.val_loss_min = val_loss