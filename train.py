import os
import time
import torch
import numpy as np
from models.model import Model
from datasets.data_loader import load_dataset, load_graph_data
from utils.earlystopping import EarlyStopping
from utils.metrices import metric, masked_mae_loss


class Training():
    def __init__(self, args, file_logger):
        self.args = args

        self.adj, self.in_degree, self.out_degree = load_graph_data(os.path.join('datasets', args.dataset, 'graph_data.pkl'))
        self.adj = torch.from_numpy(self.adj).to(args.device)
        self.in_degree = self.in_degree.to(args.device)
        self.out_degree = self.out_degree.to(args.device)

        self.load_dataset = load_dataset
        self.file_logger = file_logger
        # 模型
        self.model = Model(self.args).to(args.device)
        # 损失函数
        self.criterion = masked_mae_loss
        # Adam优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate,
                                          weight_decay=args.weight_decay, eps=args.eps)
        # learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_sche_steps,
                                                                 gamma=args.lr_decay_ratio) if args.if_lr_scheduler else None

    def train(self):
        # 训练集、验证集数据加载
        train_loader, train_scaler = self.load_dataset(os.path.join('./datasets', self.args.dataset, str(self.args.seq_len)),
                                                       self.args.batch_size, 'train')
        valid_loader, valid_scaler = self.load_dataset(os.path.join('./datasets', self.args.dataset, str(self.args.seq_len)),
                                                       self.args.batch_size, 'val')

        print("Whole trainining iteration is " + str(train_loader.num_batch))
        early_stopping = EarlyStopping(self.args.patience, self.args.save_path, self.file_logger)
        train_time, val_time = [], []
        batch_num = 0

        # ========================== Train ==========================
        for epoch in range(self.args.epochs):
            start_time = time.time()
            train_loss, train_mae, train_rmse = [], [], []
            for i, (batch_x, batch_y) in enumerate(train_loader.get_iterator()):
                self.model.train()
                self.optimizer.zero_grad()

                # PEMS-BAY数据集296特征 构造时间戳信息
                if self.args.input_dim == 296 and self.args.dataset == 'PEMS-BAY':
                    B, T, N, _ = batch_x.shape
                    time_of_day = []
                    for b in range(B):
                        tmp = np.eye(288, dtype=np.float32)[batch_x[b, :, 0, 1].astype(np.int32)]
                        tmp = np.tile(tmp, (N, 1, 1)).transpose((1, 0, 2))
                        time_of_day.append(tmp)
                    time_of_day = np.stack(time_of_day, axis=0)
                    batch_x = np.concatenate([batch_x[:, :, :, 0:1], time_of_day, batch_x[:, :, :, 2:]], axis=-1)

                batch_x = torch.tensor(batch_x).float().to(self.args.device)
                batch_y = torch.tensor(batch_y).float().to(self.args.device)
                output = self.model(batch_x, self.adj, self.in_degree, self.out_degree)

                predict = train_scaler.inverse_transform(output)
                ground_truth = train_scaler.inverse_transform(batch_y)

                loss = self.criterion(predict, ground_truth)  # 计算损失
                loss.backward()

                # 梯度裁剪
                if self.args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                mae, rmse = metric(predict, ground_truth)
                train_loss.append(loss.item())
                train_mae.append(mae)
                train_rmse.append(rmse)
                print('{}: {}'.format(i, mae), end='\r', flush=True)

                batch_num += 1

            end_time = time.time() - start_time
            train_time.append(end_time - start_time)
            cur_lr = self.optimizer.param_groups[0]['lr']

            if self.lr_scheduler:
                self.lr_scheduler.step()

            mtrain_loss = np.mean(train_loss)
            mtrain_mae = np.mean(train_mae)
            mtrain_rmse = np.mean(train_rmse)

            # ========================== Valid ==========================
            valid_start_time = time.time()
            mvalid_loss, mvalid_mae, mvalid_rmse = self.valid(valid_loader, valid_scaler)
            valid_end_time = time.time()
            val_time.append(valid_end_time - valid_start_time)

            self.file_logger.info(' | Epoch: {:03d} | Train_Loss: {:.4f} | Train_MAE: {:.4f} | Train_RMSE: {:.4f} | Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} | Valid_MAE: {:.4f} | LR: {:.6f}'.format(
                epoch, mtrain_loss, mtrain_mae, mtrain_rmse, mvalid_loss, mvalid_rmse, mvalid_mae, cur_lr))

            # Early Stopping
            early_stopping(mvalid_loss, self.model)
            if early_stopping.early_stop:
                self.file_logger.info('Early Stopping !!')
                break

        # ========================== Test ==========================
        del train_loader, train_scaler, valid_loader, valid_scaler
        test_loader, test_scaler = self.load_dataset(os.path.join('./datasets', self.args.dataset, str(self.args.seq_len)),
                                                     self.args.batch_size, 'test')
        self.model.load_state_dict(torch.load(self.args.save_path))
        self.test(self.model, self.adj, self.in_degree, self.out_degree, test_loader, test_scaler, self.args, self.file_logger)

    def valid(self, valid_loader, valid_scaler):
        valid_loss, valid_mae, valid_rmse = [], [], []
        self.model.eval()

        for i, (batch_x, batch_y) in enumerate(valid_loader.get_iterator()):
            # PEMS-BAY数据集296特征 构造时间戳信息
            if self.args.input_dim == 296 and self.args.dataset == 'PEMS-BAY':
                B, T, N, _ = batch_x.shape
                time_of_day = []
                for b in range(B):
                    tmp = np.eye(288, dtype=np.float32)[batch_x[b, :, 0, 1].astype(np.int32)]
                    tmp = np.tile(tmp, (N, 1, 1)).transpose((1, 0, 2))
                    time_of_day.append(tmp)
                time_of_day = np.stack(time_of_day, axis=0)
                batch_x = np.concatenate([batch_x[:, :, :, 0:1], time_of_day, batch_x[:, :, :, 2:]], axis=-1)

            batch_x = torch.tensor(batch_x).float().to(self.args.device)
            batch_y = torch.tensor(batch_y).float().to(self.args.device)
            output = self.model(batch_x, self.adj, self.in_degree, self.out_degree)

            predict = valid_scaler.inverse_transform(output)
            ground_truth = valid_scaler.inverse_transform(batch_y)
            loss = self.criterion(predict, ground_truth)
            mae, rmse = metric(predict, ground_truth)

            print("validation: {}".format(mae), end='\r', flush=True)

            valid_loss.append(loss.item())
            valid_mae.append(mae)
            valid_rmse.append(rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_rmse = np.mean(valid_rmse)

        return mvalid_loss, mvalid_mae, mvalid_rmse

    @staticmethod
    def test(model, adj, in_degree, out_degree, test_loader, test_scaler, params, file_logger):
        file_logger.info('Begin testing ...')
        model.eval()
        predict = []
        ground_truth = []
        for i, (batch_x, batch_y) in enumerate(test_loader.get_iterator()):
            # PEMS-BAY数据集296特征 构造时间戳信息
            if params.input_dim == 296 and params.dataset == 'PEMS-BAY':
                B, T, N, _ = batch_x.shape
                time_of_day = []
                for b in range(B):
                    tmp = np.eye(288, dtype=np.float32)[batch_x[b, :, 0, 1].astype(np.int32)]
                    tmp = np.tile(tmp, (N, 1, 1)).transpose((1, 0, 2))
                    time_of_day.append(tmp)
                time_of_day = np.stack(time_of_day, axis=0)
                batch_x = np.concatenate([batch_x[:, :, :, 0:1], time_of_day, batch_x[:, :, :, 2:]], axis=-1)

            batch_x = torch.tensor(batch_x).float().to(params.device)
            batch_y = torch.tensor(batch_y).float().to(params.device)
            output = model(batch_x, adj, in_degree, out_degree)

            output = test_scaler.inverse_transform(output)
            batch_y = test_scaler.inverse_transform(batch_y)

            predict.append(output.detach())
            ground_truth.append(batch_y.detach())

        predict = torch.cat(predict, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)

        # print('saving...')
        # np.save('predictions' + str(params.seq_len) + '.npy', [predict.detach().cpu().numpy(), ground_truth.detach().cpu().numpy()])

        mae, rmse = metric(predict, ground_truth)
        file_logger.info('(On average over {} horizons) Test MAE: {:.2f} | Test RMSE: {:.2f}'.format(params.seq_len, mae, rmse))

        if params.show_step_err:
            for i in range(params.seq_len):
                pred = predict[:, i, :, :]
                real = ground_truth[:, i, :, :]
                step_mae, step_rmse = metric(pred, real)
                file_logger.info('Horizon {}: Test MAE: {:.2f} | Test RMSE: {:.2f}'.format(i, step_mae, step_rmse))

