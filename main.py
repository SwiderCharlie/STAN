import time
import logging
import torch
from utils.utils import args, show_params
from train import Training
from models.model import Model

import os
os.environ["KMP_DUPLICATE_LIB_OK”"] = "TRUE"


def main():
    # 相关参数
    params = args()

    # log文件
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    file_logger = logging.getLogger('FileLogger')
    # 向文件输出日志信息
    file_handler = logging.FileHandler(params.logfile, mode='a', encoding='utf-8')
    file_logger.addHandler(file_handler)

    # 打印参数
    show_params(params, file_logger)

    start_time = time.time()

    if params.only_test:  # 仅预测
        from datasets.data_loader import load_graph_data, load_dataset
        model = Model(params).to(params.device)
        params.test_best_model_file = 'output/PEMS08/PEMS08-12@20230304-11-44-40-CE.pt'
        model.load_state_dict(torch.load(params.test_best_model_file), False)
        adj, _, _, _, in_degree, out_degree = load_graph_data('datasets/PEMS08/graph_data.pkl')
        adj = torch.from_numpy(adj).to(params.device)
        in_degree = in_degree.to(params.device)
        out_degree = out_degree.to(params.device)
        trainer = Training(params, file_logger)
        test_loader, test_scaler = load_dataset('datasets/PEMS08', params.batch_size, 'test')

        trainer.test(model, adj, None, None, None, in_degree, out_degree, test_loader, test_scaler, params, file_logger)

    else:  # 训练 + 预测
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        trainer = Training(params, file_logger)
        file_logger.info('Begin training ...')
        trainer.train()

    total_time = int(time.time() - start_time)
    file_logger.info("Total Time：{}:{:0>2d}:{:0>2d}".format(total_time // 3600, total_time % 3600 // 60, total_time % 60))


if __name__ == '__main__':
    main()
