import os
import shutil
import sys

import pandas as pd
from tqdm import tqdm

from ta_cn.utils_wide import WArr
os.environ['TA_CN_MODE'] = 'WIDE'
import ta_cn.alphas.alpha191 as w
from ta_cn.imports.gtja_wide import DELAY, MAX, IF

import subprocess
from datetime import datetime, timedelta

from utils.log_util import get_logger
logger = get_logger()


def to_wide_df(wide_df, symbol, data_df, column):
    if wide_df.empty:
        wide_df[symbol] = data_df[column]
    else:
        temp_df = data_df[column].rename(symbol)
        wide_df = wide_df.join(temp_df)
    return wide_df


# DTM (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
# DBM (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
def compute_dtm_dbm(OPEN, HIGH, LOW, **kwargs):
    dtm = IF(OPEN<=DELAY(OPEN,1),0,MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
    dbm = IF(OPEN>=DELAY(OPEN,1),0,MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    return dtm, dbm


def compute_alpha(kwargs_w, row_names, col_names):
    df_list = []
    name_list = []
    logger.info("computing alpha191")
    for i in tqdm(range(1, 191 + 1)):
        name = f'alpha_{i:03d}'
        # 165 183 是MAX 与 SUMAC 问题
        if i in (165, 183, 30):
            logger.warning(f"skipping {name}")
            continue
        fw = getattr(w, name, None)
        rw = fw(**kwargs_w)

        temp_df = pd.DataFrame(rw.raw(), index=row_names, columns=col_names)
        df_list.append(temp_df)
        name_list.append(name)
    alpha_df = pd.concat(df_list, keys=name_list)
    return alpha_df


def alpha_to_csv(alpha_df, src_data_dir, dst_data_dir):
    os.makedirs(dst_data_dir, exist_ok=True)
    logger.info(f"saving alpha data to csv files in {dst_data_dir}.")
    for col in tqdm(alpha_df.columns):
        csv_fn = os.path.expanduser(os.path.join(src_data_dir, f"{col}.csv"))
        df = pd.read_csv(csv_fn)
        df.set_index("date", inplace=True)
        ins_alpha = alpha_df[col].unstack(level=0)
        out_df = df.join(ins_alpha)
        out_df.reset_index(inplace=True)
        out_csv_fn = os.path.expanduser(os.path.join(dst_data_dir, f"{col}.csv"))
        out_df.to_csv(out_csv_fn, index=False)


def get_benchmark_file(src_data_dir, benchmark="SH000300"):
    benchmark_file = os.path.expanduser(os.path.join(src_data_dir, f"{benchmark.upper()}.csv"))
    if not os.path.exists(benchmark_file):
        benchmark_file = os.path.expanduser(os.path.join(src_data_dir, f"{benchmark.lower()}.csv"))
    logger.info(f"benchmark file {benchmark_file}")
    return benchmark_file


def prepare_wide_args(src_data_dir, instruments_file, benchmark_file):
    logger.info("preparing arguments in wide table format.")
    with open(instruments_file, "r") as f:
        instruments = f.readlines()
    instruments = [elem.split("\t")[0] for elem in instruments]
    data_files = [(elem, os.path.join(src_data_dir, f"{elem}.csv")) for elem in instruments]

    open_ = pd.DataFrame()
    high = pd.DataFrame()
    low = pd.DataFrame()
    close = pd.DataFrame()
    volume = pd.DataFrame()
    amount = pd.DataFrame()
    vwap = pd.DataFrame()
    returns = pd.DataFrame()
    bm_open = pd.DataFrame()
    bm_close = pd.DataFrame()

    bm_df = pd.read_csv(benchmark_file)
    bm_df.set_index("date", inplace=True)

    logger.info(f"processing instrument files in {src_data_dir}.")

    for elem in tqdm(data_files):
        symbol, fn = elem
        df = pd.read_csv(fn)
        df.set_index("date", inplace=True)
        df["vwap"] = df["amount"] / df["volume"]
        open_ = to_wide_df(open_, symbol, df, "open")
        high = to_wide_df(high, symbol, df, "high")
        low = to_wide_df(low, symbol, df, "low")
        close = to_wide_df(close, symbol, df, "close")
        amount = to_wide_df(amount, symbol, df, "amount")
        volume = to_wide_df(volume, symbol, df, "volume")
        vwap = to_wide_df(vwap, symbol, df, "vwap")
        returns = to_wide_df(returns, symbol, df, "pctChg")

        bm_open = to_wide_df(bm_open, symbol, bm_df, "open")
        bm_close = to_wide_df(bm_close, symbol, bm_df, "close")


    volume = volume.astype(float)
    bm_open = bm_open.loc[bm_open.index.intersection(high.index)]
    bm_close = bm_close.loc[bm_close.index.intersection(high.index)]

    row_names = high.index
    col_names = high.columns

    logger.info(f"open shape {open_.shape}, high shape {high.shape}, close shape {close.shape}, returns shape {returns.shape}")
    logger.info(f"volume shape {volume.shape}, bm_open shape {bm_open.shape}, bm_close shape {bm_close.shape}")

    # OPEN 开盘价
    # HIGH 最高价
    # LOW 最低价
    # CLOSE 收盘价
    # VWAP 均价
    # VOLUME 成交量
    # AMOUNT 成交额
    # BANCHMARKINDEXCLOSE 基准指数的开盘价
    # BANCHMARKINDEXOPEN 基准指数的收盘价
    # RET 每日收益率(收盘/前收盘-1)
    # DTM (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
    # DBM (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    # HML SMB MKE Fama French 三因子

    df = {
        "OPEN": open_,
        "HIGH": high,
        "LOW": low,
        "CLOSE": close,
        "VWAP": vwap,
        "VOLUME": volume,
        "AMOUNT": amount,
        "BANCHMARKINDEXOPEN": bm_open,
        "BANCHMARKINDEXCLOSE": bm_close,
        "RET": returns,
    }

    kwargs_w = {k: WArr.from_array(v, direction='down') for k, v in df.items()}
    logger.info("kwargs_w {}".format(kwargs_w))
    dtm, dbm = compute_dtm_dbm(**kwargs_w)
    kwargs_w["DTM"] = dtm
    kwargs_w["DBM"] = dbm

    return kwargs_w, row_names, col_names


def generate_alpha191_dirs(ds, stock_data_dir, qlib_data_dir):
    stock_data_dir = stock_data_dir.replace(ds, "").replace("__", "_").strip("_")
    qlib_data_dir = qlib_data_dir.replace(ds, "").replace("__", "_").strip("_")
    alpha191_data_dir = f"{stock_data_dir}_alpha191_{ds}"
    alpha191_qlib_dir = f"{qlib_data_dir}_alpha191_{ds}"
    return alpha191_data_dir, alpha191_qlib_dir


# stock_data_dir="~/.qlib/stock_data/source/bao_cn_data_20240104",
# alpha191_data_dir="~/.qlib/stock_data/source/bao_cn_data_20240104_alpha191",
# alpha191_qlib_dir="~/.qlib/qlib_data/bao_cn_data_20240104_alpha191",
# instruments_file="~/.qlib/qlib_data/bao_cn_data_20240104/instruments/all.txt",
# benchmark="SH000300"
def stock_data_to_alpha191(ds, stock_data_dir, qlib_data_dir, instruments_file, benchmark):
    alpha191_data_dir, alpha191_qlib_dir = generate_alpha191_dirs(ds, stock_data_dir, qlib_data_dir)
    alpha191_data_dir = os.path.expanduser(alpha191_data_dir)
    alpha191_qlib_dir = os.path.expanduser(alpha191_qlib_dir)
    instruments_file = os.path.expanduser(instruments_file)
    benchmark_file = get_benchmark_file(stock_data_dir, benchmark)

    kwargs_w, row_names, col_names = prepare_wide_args(src_data_dir=stock_data_dir, instruments_file=instruments_file,
                                                       benchmark_file=benchmark_file)
    logger.info(f"row_names {row_names.shape}, col_names {col_names.shape}")
    alpha_df = compute_alpha(kwargs_w, row_names, col_names)
    alpha_to_csv(alpha_df, src_data_dir=stock_data_dir, dst_data_dir=alpha191_data_dir)
    shutil.copy(benchmark_file, f"{alpha191_data_dir}/{benchmark}.csv")
    logger.info("dump alpha191 data")
    alpha191_to_bin(alpha191_data_dir, alpha191_qlib_dir)
    logger.info("clean up old data")
    cleanup_old_files(ds, stock_data_dir, qlib_data_dir)
    alpha191_instruments_file = f"{alpha191_data_dir}/instruments/bao_filter.txt"
    return alpha191_data_dir, alpha191_qlib_dir, alpha191_instruments_file


def alpha191_to_bin(alpha191_data_dir, alpha191_qlib_dir):
    command = f"python scripts/dump_bin.py dump_all --csv_path {alpha191_data_dir} --qlib_dir {alpha191_qlib_dir} --freq day --exclude_fields date,symbol"
    subprocess.run(command, shell=True)

    command = f"python scripts/data_collector/baostock_1d/collector.py remove_index_instuments --instrments_dir={alpha191_qlib_dir}/instruments"
    subprocess.run(command, shell=True)


def cleanup_old_files(ds, stock_data_dir, qlib_data_dir):
    for i in range(3, 10):
        temp_ds = datetime.strptime(ds, "%Y%m%d") - timedelta(days=i)
        temp_ds = temp_ds.strftime("%Y%m%d")
        alpha191_data_dir, alpha191_qlib_dir = generate_alpha191_dirs(temp_ds, stock_data_dir, qlib_data_dir)
        logger.info(f"removing {alpha191_data_dir} and {alpha191_qlib_dir}")
        shutil.rmtree(alpha191_data_dir, ignore_errors=True)
        shutil.rmtree(alpha191_qlib_dir, ignore_errors=True)
