"""Trainer for RhythmMamba_fusion_scan."""
import os, numpy as np, torch, torch.optim as optim, random
from tqdm import tqdm
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.RhythmMamba_fusion_scan import RhythmMamba_fusion_scan
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss


class RhythmMambaFusionScanTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        self.data_dict = {}

        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1

        self.grid_size = getattr(config.MODEL, 'GRID_SIZE', 3)
        self.modulation_mode = getattr(config.MODEL, 'MODULATION_MODE', 'scalar')
        self.quality_scale_init = getattr(config.MODEL, 'QUALITY_SCALE_INIT', 0.1)
        self.quality_mode = getattr(config.MODEL, 'QUALITY_MODE', 'fusion')
        self.osc_learn_omega = getattr(config.MODEL, 'OSC_LEARN_OMEGA', True)

        if config.TOOLBOX_MODE in ("train_and_test", "only_test"):
            self.model = RhythmMamba_fusion_scan(
                depth=24, embed_dim=96, mlp_ratio=2, grid_size=self.grid_size,
                modulation_mode=self.modulation_mode,
                quality_scale_init=self.quality_scale_init,
                quality_mode=self.quality_mode,
                osc_learn_omega=self.osc_learn_omega,
            ).to(self.device)
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            if config.TOOLBOX_MODE == "train_and_test":
                self.num_train_batches = len(data_loader["train"])
                self.criterion = Hybrid_Loss()
                self.optimizer = optim.AdamW(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                    steps_per_epoch=self.num_train_batches)
        else:
            raise ValueError("Invalid toolbox mode!")

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        for epoch in range(self.max_epoch_num):
            print(f'\n====Training Epoch: {epoch}====')
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Train epoch {epoch}")
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape
                if self.config.TRAIN.AUG:
                    data, labels = self.data_augmentation(data, labels, batch[2], batch[3])
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                pred_ppg = (pred_ppg - pred_ppg.mean(-1, keepdim=True)) / (pred_ppg.std(-1, keepdim=True) + 1e-8)
                loss = sum(self.criterion(pred_ppg[ib], labels[ib], epoch, self.config.TRAIN.DATA.FS, self.diff_flag)
                          for ib in range(N)) / N
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                vloss = self.valid(data_loader)
                print(f'validation loss: {vloss}')
                if self.min_valid_loss is None or vloss < self.min_valid_loss:
                    self.min_valid_loss = vloss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")
        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"best epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            return float('inf')
        print('\n===Validating===')
        self.model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in tqdm(data_loader["valid"], ncols=80, desc="Validation"):
                data_v, labels_v = batch[0].to(self.device), batch[1].to(self.device)
                pred_v = self.model(data_v)
                pred_v = (pred_v - pred_v.mean(-1, keepdim=True)) / (pred_v.std(-1, keepdim=True) + 1e-8)
                for ib in range(data_v.shape[0]):
                    valid_losses.append(self.criterion(pred_v[ib], labels_v[ib],
                                        self.config.TRAIN.EPOCHS, self.config.VALID.DATA.FS,
                                        self.diff_flag).item())
        return np.mean(valid_losses)

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print('\n===Testing===')
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("MODEL_PATH error!")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        else:
            ep = self.max_epoch_num - 1 if self.config.TEST.USE_LAST_EPOCH else self.best_epoch
            ckpt = os.path.join(self.model_dir, f'{self.model_file_name}_Epoch{ep}.pth')
            print(f"Loading: {ckpt}")
            self.model.load_state_dict(torch.load(ckpt))
        self.model.eval()
        with torch.no_grad():
            predictions, labels_dict = dict(), dict()
            for _, batch in enumerate(data_loader['test']):
                bs = batch[0].shape[0]
                data_t, labels_t = batch[0].to(self.config.DEVICE), batch[1].to(self.config.DEVICE)
                pred_t = self.model(data_t)
                pred_t = (pred_t - pred_t.mean(-1, keepdim=True)) / (pred_t.std(-1, keepdim=True) + 1e-8)
                labels_t = labels_t.view(-1, 1)
                pred_t = pred_t.view(-1, 1)
                for ib in range(bs):
                    sid, si = batch[2][ib], int(batch[3][ib])
                    predictions.setdefault(sid, {})[si] = pred_t[ib * self.chunk_len:(ib + 1) * self.chunk_len]
                    labels_dict.setdefault(sid, {})[si] = labels_t[ib * self.chunk_len:(ib + 1) * self.chunk_len]
            print(' ')
            calculate_metrics(predictions, labels_dict, self.config)

    def save_model(self, index):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f'{self.model_file_name}_Epoch{index}.pth')
        torch.save(self.model.state_dict(), path)
        print(f'Saved: {path}')

    def data_augmentation(self, data, labels, index1, index2):
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W)); labels_aug = np.zeros((N, D))
        for idx in range(N):
            index = index1[idx] + index2[idx]
            if np.random.random() < 0.5:
                if index in self.data_dict:
                    gt_hr_fft = self.data_dict[index]
                else:
                    gt_hr_fft, _ = calculate_hr(labels[idx], labels[idx], diff_flag=self.diff_flag,
                                                 fs=self.config.VALID.DATA.FS)
                    self.data_dict[index] = gt_hr_fft
                if gt_hr_fft > 90:
                    ei = torch.arange(0, D, 2); oi = ei + 1
                    r3 = random.randint(0, D // 2 - 1)
                    data_aug[:, ei] = data[:, r3 + ei // 2]; labels_aug[:, ei] = labels[:, r3 + ei // 2]
                    data_aug[:, oi] = (data[:, r3 + oi // 2] + data[:, r3 + oi // 2 + 1]) / 2
                    labels_aug[:, oi] = (labels[:, r3 + oi // 2] + labels[:, r3 + oi // 2 + 1]) / 2
                elif gt_hr_fft < 75:
                    data_aug[:, :D // 2] = data[:, ::2]; labels_aug[:, :D // 2] = labels[:, ::2]
                    data_aug[:, D // 2:] = data_aug[:, :D // 2]; labels_aug[:, D // 2:] = labels_aug[:, :D // 2]
                else:
                    data_aug[idx], labels_aug[idx] = data[idx], labels[idx]
            else:
                data_aug[idx], labels_aug[idx] = data[idx], labels[idx]
        data_aug = torch.tensor(data_aug).float(); labels_aug = torch.tensor(labels_aug).float()
        if np.random.random() < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        return data_aug, labels_aug
