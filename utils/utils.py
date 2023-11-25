import argparse
import torch
import os
import time


def show_params(args, file_logger):
    args_dic = vars(args)
    file_logger.info("--------------------------------------------")
    for key, value in args_dic.items():
        file_logger.info('|%20s:|%40s|' % (key, value))
    file_logger.info("--------------------------------------------")


def args():
    # 相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PEMS08')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu index')
    parser.add_argument('--logfile', type=str, default='.\log', help="location of models checkpoints")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-04)
    parser.add_argument('--eps', type=float, default=1e-08)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--if_cl', type=bool, default=False)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--if_lr_scheduler', type=bool, default=False)
    parser.add_argument('--lr_sche_steps', type=list, default=[1, 30, 38, 46, 54, 200])
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--show_step_err', type=bool, default=True)
    parser.add_argument('--only_test', type=bool, default=True)
    # parser.add_argument('--test_best_model_file', type=str, default='xxx.pt')
    parser.add_argument('--input_dim', type=int, default=296)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--feedforward_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--num_nodes', type=int, default=170)
    parser.add_argument('--no_meta_temporal', type=bool, default=False)
    parser.add_argument('--no_meta_spatial', type=bool, default=False)
    parser.add_argument('--num_groups', type=int, default=4)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--cl_epochs', type=int, default=3)
    parser.add_argument('--if_topk', type=bool, default=False)
    parser.add_argument('--topk', type=int, default=30)

    params = parser.parse_args()

    params.use_gpu = True if torch.cuda.is_available() and params.use_gpu else False
    params.device = torch.device('cuda:{}'.format(params.gpu)) if params.use_gpu else 'cpu'
    cur_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
    logfilename = params.dataset + '-' + str(params.seq_len) + '@' + cur_time
    params.logfile = os.path.join(params.logfile, logfilename)
    params.save_path = os.path.join('.\output', params.dataset, logfilename + '.pt')
    # params.test_best_model_file = os.path.join('./output', params.dataset, params.test_best_model_file)
    if params.dataset == 'PEMS-BAY':
        params.num_nodes = 325
    elif params.dataset == 'PEMS08':
        params.num_nodes = 170

    return params


