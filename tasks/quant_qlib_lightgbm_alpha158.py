import os, sys
import traceback

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(f"parent dir {parent_dir}, cwd {os.getcwd()}")
sys.path.append(parent_dir)

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data.dataset.handler import DataHandlerLP


from scripts.data_collector.utils import get_calendar_list

from tasks.alpha191 import stock_data_to_alpha191

from utils.log_util import get_logger
logger = get_logger()


from datetime import datetime, timedelta
from ruamel.yaml import YAML

import subprocess
import pandas as pd

import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from utils.config_util import get_mlflow_tracking_config
from utils.metric_util import compute_precision_recall, save_prediction_to_db


from functools import partial


PRED_LABEL_NAME = "label_roi_5d"

def record_to_db(ds, model_name, recorder_id):
    recorder = R.get_recorder(recorder_id=recorder_id)
    pred_df = recorder.load_object("pred.pkl")
    label_df = recorder.load_object("label.pkl")
    df = pred_df.join(label_df, how="inner")
    df = df.reset_index()
    df.columns = ["日期", "代码", "score", "label"]
    df = df.dropna(subset=["score", "label"])
    df['代码'] = df['代码'].str.replace('SH|SZ', '', regex=True)
    df["score"] = df["score"] * 100
    df["label"] = df["label"] * 100

    pred = df["score"].values.flatten()
    label = df["label"].values.flatten()
    compute_precision_recall(label, pred)

    save_prediction_to_db(df, ds=ds, model_name=model_name, stage="validation", pred_name=PRED_LABEL_NAME,
                          pred_val_col_name="score", label_col_name="label", run_id=recorder_id)

    online_pred_df = pd.DataFrame(recorder.load_object("online_pred.pkl"))
    online_pred_df = online_pred_df.reset_index()
    online_pred_df["label"] = 0

    online_pred_df.columns = ["日期", "代码", "score", "label"]
    online_pred_df['代码'] = online_pred_df['代码'].str.replace('SH|SZ', '', regex=True)
    online_pred_df["score"] = online_pred_df["score"] * 100


    save_prediction_to_db(online_pred_df, ds=ds, model_name=model_name, stage="prediction", pred_name=PRED_LABEL_NAME,
                          pred_val_col_name="score", label_col_name="label", run_id=recorder_id)




def download_bao_data(ds):
    train_start, train_end, valid_start, valid_end, test_start, test_end, pred_start, pred_end = get_dates(ds)
    script_path = 'scripts/get_bao_data.sh'
    stock_data_dir_prefix = "~/.qlib/stock_data/source/bao_cn_data"
    qlib_data_dir_prefix = "~/.qlib/qlib_data/bao_cn_data"

    subprocess.call(['sh', script_path, ds, test_end, pred_end, stock_data_dir_prefix, qlib_data_dir_prefix])
    stock_bao_data = f"~/.qlib/stock_data/source/bao_cn_data_{ds}"
    qlib_bao_data = f"~/.qlib/qlib_data/bao_cn_data_{ds}"
    instruments_file = f"~/.qlib/qlib_data/bao_cn_data_{ds}/instruments/filter.txt"
    return stock_bao_data, qlib_bao_data, instruments_file


def download_db_data(ds):
    stock_data_dir_prefix = "~/.qlib/stock_data/source/db_data"
    qlib_data_dir_prefix = "~/.qlib/qlib_data/db_data"
    stock_data_dir = os.path.expanduser(f"{stock_data_dir_prefix}_{ds}")
    qlib_data_dir = os.path.expanduser(f"{qlib_data_dir_prefix}_{ds}")
    instruments_file = f"{qlib_data_dir}/instruments/filter.txt"

    if os.path.exists(stock_data_dir):
        logger.info(f"data already downloaded. Skip downloading. Delete folder {stock_data_dir} to download again.")
        return stock_data_dir, qlib_data_dir, instruments_file

    train_start, train_end, valid_start, valid_end, test_start, test_end, pred_start, pred_end = get_dates(ds)
    os.makedirs(stock_data_dir, exist_ok=True)
    os.makedirs(qlib_data_dir, exist_ok=True)

    download_csv(stock_data_dir, train_start, pred_end)
    csv_to_bin(stock_data_dir, qlib_data_dir, instruments_file)
    clean_up_old_files(ds, stock_data_dir_prefix, qlib_data_dir_prefix)

    return stock_data_dir, qlib_data_dir, instruments_file


def download_csv(stock_data_dir, start, end, collector="db"):
    download_cmd = f"python scripts/data_collector/database_1d/collector.py download_data --source_dir {stock_data_dir} --start {start} --end {end} --interval 1d --region CN --max_workers 16"
    if collector == "bao":
        download_cmd = f"python scripts/data_collector/baostock_1d/collector.py download_data --source_dir {stock_data_dir} --start {start} --end {end} --interval 1d --region CN"
    subprocess.run(download_cmd, shell=True)


def csv_to_bin(stock_data_dir, qlib_data_dir, instruments_file):
    # dump
    logger.info("dump data into qlib bin format")
    dump_cmd = f"python scripts/dump_bin.py dump_all --csv_path {stock_data_dir} --qlib_dir {qlib_data_dir} --freq day --exclude_fields date,symbol"
    subprocess.run(dump_cmd, shell=True)
    # generate new instrument file
    logger.info("generate new instrument file")
    ins_cmd = f"python scripts/dump_bin.py remove_index_instuments --instrments_dir={qlib_data_dir}/instruments --instruments_file={instruments_file}"
    subprocess.run(ins_cmd, shell=True)


def clean_up_old_files(ds, stock_data_dir_prefix, qlib_data_dir_prefix):
    import shutil
    for i in range(3, 10):
        temp_ds = datetime.strptime(ds, "%Y%m%d") - timedelta(days=i)
        temp_ds = temp_ds.strftime("%Y%m%d")
        data_dir, qlib_dir = f"{stock_data_dir_prefix}_{temp_ds}", f"{qlib_data_dir_prefix}_{temp_ds}"
        logger.info(f"removing {data_dir} and {qlib_dir}")
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(qlib_dir, ignore_errors=True)


def get_dates(ds):
    ds = datetime.strptime(ds, "%Y%m%d")
    train_start = ds - timedelta(days=365 * 2)
    train_end = ds - timedelta(days=61)
    valid_start = ds - timedelta(days=60)
    valid_end = ds - timedelta(days=31)
    test_start = ds - timedelta(days=30)
    test_end = ds - timedelta(days=2)
    pred_start = ds - timedelta(days=1)
    pred_end = ds - timedelta(days=-1)
    res = [train_start, train_end, valid_start, valid_end, test_start, test_end, pred_start, pred_end]
    logger.info(f"train_start {train_start}, train_end {train_end}, valid_start {valid_start}, valid_end {valid_end}, test_start {test_start}, test_end {test_end}, pred_start {pred_start}, pred_end {pred_end}")
    return (elem.strftime("%Y-%m-%d") for elem in res)



def generate_config_file(ds, template_path, config_path, qlib_data_path):
    train_start, train_end, valid_start, valid_end, test_start, test_end, pred_start, pred_end = get_dates(ds)
    with open(template_path, "r") as fr, open(config_path, "w") as fw:
        content = fr.read()
        content = content.replace("$bao_cn_data", qlib_data_path)
        content = content.replace("$train_start", train_start)
        content = content.replace("$train_end", train_end)
        content = content.replace("$valid_start", valid_start)
        content = content.replace("$valid_end", valid_end)
        content = content.replace("$test_start", test_start)
        content = content.replace("$test_end", test_end)
        content = content.replace("$pred_start", pred_start)
        content = content.replace("$pred_end", pred_end)

        fw.write(content)
    logger.info(f"configuration written to {config_path}")


def get_config_from_file(config_path):
    with open(config_path) as fp:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(fp)
    return config


def generate_exp_name(config):
    task = config["task"]
    model_name = task["model"]["class"].lower()
    dh_name = task["dataset"]["kwargs"]["handler"]["class"].lower()
    return f"qlib_bao_{dh_name}_{model_name}"

def generate_recorder_name(ds):
    return str(ds)

def get_mlflow_exp_manager():
    server_address, port, user, password = get_mlflow_tracking_config()
    import os
    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    exp_manager = {
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": f"http://{server_address}:{port}",
            "default_exp_name": "Experiment",
        }
    }
    return exp_manager

def init_qlib(config):
    provider_uri = config["qlib_init"]["provider_uri"]
    exp_manager = get_mlflow_exp_manager()
    qlib.init(provider_uri=provider_uri, region=REG_CN, exp_manager=exp_manager)
    logger.info(f"qlib initialised with uri {provider_uri}, region {REG_CN}")

def train_and_predict(config, ds):
    task = config["task"]
    logger.info(f"task configuration {task}")

    dataset = init_instance_by_config(task["dataset"])
    logger.info(f"dataset {dataset}")
    # train = dataset.prepare(segments='train', col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    # print(f"train {train}")
    # nan_columns = train["feature"].columns[train["feature"].isnull().all()].tolist()
    # logger.info(f"NOTE: NAN features {nan_columns}, all features {train['feature'].columns.tolist()}")

    model = init_instance_by_config(task["model"])
    logger.info(f"model {model}")

    exp_name = generate_exp_name(config)
    recorder_name = generate_recorder_name(ds)
    MODEL_FILE_NAME = "params.pkl"

    # start exp
    with R.start(experiment_name=exp_name, recorder_name=recorder_name):
        logger.info(f"experiment {exp_name}, recorder name {recorder_name}, rid {R.get_recorder().id}, model file name {MODEL_FILE_NAME}")

        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(**{MODEL_FILE_NAME: model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        port_analysis_config = config["port_analysis_config"]
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

        # load previous results
        # pred_df = recorder.load_object("pred.pkl")
        # report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        # positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        # analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

        # analysis_position.report_graph(report_normal_df)

        logger.info("now making predictions")
        df = model.predict(dataset=dataset, segment="pred")
        if isinstance(df, pd.Series):
            df = df.to_frame("score")
        recorder.save_objects(**{"online_pred.pkl": df})
        # logger.info("online prediction saved to online_pred.pkl. the following code could be used to read it.")
        # logger.info(f"import qlib; qlib.init(provider_uri='{provider_uri}', region='{REG_CN}'); from qlib.workflow import R; R.get_recorder(recorder_id='{recorder.id}').load_object('online_pred.pkl')")

        print(f"index values {df.index.levels[0]}")
        for val in df.index.levels[0]:
            temp_df = df[df.index.get_level_values(0) == val]
            if not temp_df.empty:
                logger.info(f"{temp_df.sort_values(by='score', ascending=False).head(50)*100}")
        logger.info("predictions done")

        record_to_db(ds, exp_name, recorder.id)

        # print(f"prediction result order by score {pred_df.iloc[:, 2].sort_values(ascending=False)}")


def load_optuna_data(config):
    task = config["task"]
    dataset = init_instance_by_config(task["dataset"])
    train = dataset.prepare(segments="train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    valid = dataset.prepare(segments="valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    feature_names = train["feature"].columns.tolist()
    logger.info(f"samples from train dataset {train}, train columns {train.columns}")
    # train = train.dropna(subset=["LABEL0"])
    # logger.info(f"train dataset after dropna {train}")
    # valid = valid.dropna(subset=["LABEL0"])
    return train, valid, feature_names

def train_optuna(trial, train, valid, feature_names):
    param = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        # "objective": "huber",
        # "alpha": 5.0,
        "verbosity": -1,
        "num_threads": trial.suggest_int("num_threads", 8, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 100.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 100.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 512),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 14),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 1000),
        "max_bin": trial.suggest_int("max_bin", 5, 5000),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 10000, log=True),
        "num_iterations": trial.suggest_int("num_iterations", 100, 200)
    }

    dtrain = lgb.Dataset(train["feature"], label=train["label"], feature_name=feature_names)
    dvalid = lgb.Dataset(valid["feature"], label=valid["label"], feature_name=feature_names)

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l2", valid_name="valid")

    evals_result = {}
    verbose_eval_callback = lgb.log_evaluation(period=20)
    evals_result_callback = lgb.record_evaluation(evals_result)

    lgbm = lgb.train(param, dtrain, valid_sets=[dtrain, dvalid], valid_names=["train", "valid"], callbacks=[pruning_callback, verbose_eval_callback, evals_result_callback])
    mse = mean_squared_error(valid["label"], lgbm.predict(valid["feature"]))
    print(f"mse is {mse}")
    return mse


def auto_tune(config):
    logger.info("load auto tune data")
    train, valid, feature_names = load_optuna_data(config)
    logger.info("start auto tuning")
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="minimize")
    study.optimize(partial(train_optuna, train=train, valid=valid, feature_names=feature_names), n_trials=30)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    '{}': {},".format(key, value))

    return trial.params

def run(model, factor, enable_auto_tune):
    template_config_path = f"tasks/workflow_config_{model}_{factor}_template.yaml"
    user_config_path = f"tasks/workflow_config_{model}_{factor}_{ds}.yaml"
    logger.info("generate config files")
    generate_config_file(ds, template_config_path, user_config_path, qlib_data_path=qlib_data_path)
    config = get_config_from_file(user_config_path)
    init_qlib(config)
    if enable_auto_tune:
        best_params = auto_tune(config)
        config["task"]["model"]["kwargs"] = best_params
        logger.info(f"auto tuning finished. model params: {config['task']['model']}")
    logger.info("train model and make predictions")
    train_and_predict(config, ds)


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print("ds is empty, please specify a date in the format of yyyyMMdd")
        exit(0)

    ds = sys.argv[1]
    logger.info(f"running task on ds {ds}")

    cal_list = get_calendar_list("CSI300")
    cal_list = [cal.strftime("%Y%m%d") for cal in cal_list]

    if ds not in cal_list:
        logger.info("ds not in trade calendar, task will not be executed.")
        exit(0)

    # stock_data_dir, qlib_data_dir, instruments_file = download_bao_data(ds)
    stock_data_dir, qlib_data_dir, instruments_file = download_db_data(ds)

    mode = sys.argv[2] if len(sys.argv) > 2 else "download"
    model = sys.argv[3] if len(sys.argv) > 3 else "lightgbm"
    enable_auto_tune = True if model == "lightgbm" else False

    logger.info(f"mode {mode}, model {model}, enable_auto_tune {enable_auto_tune}")

    if mode == "download":
        logger.info("download data only.")
        exit(0)
    elif mode == "alpha191":
        alpha191_data_dir, alpha191_qlib_dir, alpha191_instruments_file = \
            stock_data_to_alpha191(ds=ds,
                                   stock_data_dir=stock_data_dir,
                                   qlib_data_dir=qlib_data_dir,
                                   instruments_file=instruments_file,
                                   benchmark="SH000300")
        qlib_data_path = alpha191_qlib_dir
    else:
        qlib_data_path = qlib_data_dir

    run(model=model, factor=mode, enable_auto_tune=enable_auto_tune)
