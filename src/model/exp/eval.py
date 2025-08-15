import torch

import utils
from model.exp.acc_runtime import CONFIG_DIR, RS, ACC
from model.exp.mlogging import progress_bar

log = utils.log.get_logger()


def cal_flops(nn: torch.nn.Module, data_input_size=(1, 28, 28)):
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(nn, data_input_size,
                                                 as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def init_config():
    from model.exp import Config
    config = Config(config_file=CONFIG_DIR / f"{RS.config_path}.toml")

    if ACC.is_main_process:
        utils.output.dictionary(config.model_dump(), out_fun=log.info)
        RS.tlog.hyper_parameters(config.model_dump())

    return config


def eval_network(network: torch.nn.Module, data_loader, ):
    pass


def main():
    from model.network import net_builder
    config = init_config()
    network = net_builder(config.network_config)
    network.load_model(RS.output_dir/'src.main.debug')
    # eval_loader = config.eval_data.get_dataloader()

    cal_flops(network)
    print("done")


if __name__ == '__main__':
    main()
