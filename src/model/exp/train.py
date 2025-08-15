import utils
from model.exp.acc_runtime import CONFIG_DIR, RS, ACC
from model.exp import Config, Model
from model.exp.mlogging import progress_bar

log = utils.log.get_logger()


def init_model() -> Model:
    config = Config(config_file=CONFIG_DIR / f"{RS.config_path}.toml")

    if ACC.is_main_process:
        utils.output.dictionary(config.model_dump(), out_fun=log.info)
        RS.tlog.hyper_parameters(config.model_dump())

    model = Model(config)
    if RS.version.startswith("resume"):
        resumed_version = utils.remove_special_char(RS.version.removeprefix("resume"), mode='abc+n')
        acc_cache_dir = [
            log_dir for log_dir in (RS.output_dir / "log").iterdir()
            if (resumed_version.upper() in log_dir.name.upper()) and ('resume' not in log_dir.name)
        ]
        assert len(acc_cache_dir) == 1, f"{acc_cache_dir} should have one directory"
        ACC.load_state(acc_cache_dir[0] / 'state_cache')

        if ACC.is_main_process:
            log.warning(f"Resumed from {acc_cache_dir[0] / 'state_cache'}")

    if ACC.is_main_process:
        from l3ac import get_model_info
        codec_info = get_model_info(ACC.unwrap_model(model.network), eval_flops_seconds=10, sample_rate=model.mc.sample_rate)
        utils.output.dictionary(codec_info, out_fun=log.info)
    return model


def train():
    model = init_model()
    start_epoch, total_epoch = model.estimate_progress()
    train_with_discriminator = 'network_gen_loss' in model.mc.loss_config['loss_weights']
    for epoch in progress_bar(range(start_epoch, total_epoch), desc="Epoch"):
        if train_with_discriminator:
            model.train_epoch()
        else:
            model.train_epoch_without_discriminator()
        metric_results = model.evaluate(model.eval_loader, "evaluating")
        ACC.save_state(RS.log_path / 'state_cache')

        if ACC.is_main_process:
            log.info(f"Eval epoch({epoch}) score: {metric_results}")
        ACC.wait_for_everyone()

    if ACC.is_main_process:
        ACC.unwrap_model(model.network).save_model(RS.output_path)
        log.info(f"Finished training.")

    ACC.end_training()


if __name__ == '__main__':
    train()
