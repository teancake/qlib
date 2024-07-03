# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os.path
import sys
import copy
import traceback

import fire
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Iterable, List
import requests
import time
from datetime import datetime

import re

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent.parent))


import qlib
from qlib.data import D

from scripts.data_collector.base import BaseCollector, BaseNormalize, BaseRun
from scripts.data_collector.utils import generate_minutes_calendar_from_daily, calc_adjusted_price, get_hs_stock_symbols
from utils.starrocks_db_util import StarrocksDbUtil, query_all_stock

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"

class DatabaseCollectorCN1d(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [5min], default 5min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        super(DatabaseCollectorCN1d, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        self.data_df = pd.DataFrame()

    def download_db_data(self):
        logger.info("get history data from starrocks db.")
        sql = "select * from dwd_stock_zh_a_hist_df where ds in (select max(ds) from dwd_stock_zh_a_hist_df) and adjust='hfq' and period='daily';"
        data = StarrocksDbUtil().run_sql(sql)
        data_df = pd.DataFrame(data)
        logger.info("data obtained")
        data_df = data_df[["代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]]
        data_df = data_df.rename(
            columns={"代码": "symbol", "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                     "成交量": "volume", "成交额": "amount"})
        # adding index, very important for efficient query from data_df
        data_df = data_df.set_index(["symbol", "date"])
        data_df = self.add_vwap(data_df)
        return data_df


    def download_index_data(self):
        for _index_name, _index_code in {"csi300": "sh000300", "csi100": "sh000903", "csi500": "sh000905"}.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            sql = f'''select * from dwd_index_zh_a_hist_df 
                    where ds in (select max(ds) from dwd_index_zh_a_hist_df) 
                    and 代码='{_index_code}' and period='daily';'''
            data = StarrocksDbUtil().run_sql(sql)
            df = pd.DataFrame(data)

            df = df[["代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]]
            df = df.rename(
                columns={"代码": "symbol", "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                         "成交量": "volume", "成交额": "amount"})
            df["symbol"] = df["symbol"].str.upper()
            df["adjclose"] = df["close"]
            print(df)
            # df = df.set_index(["symbol", "date"])

            # df = df.astype(float, errors="ignore")
            _path = self.save_dir.joinpath(f"{_index_code}.csv")
            df.to_csv(_path, index=False)


    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        try:
            df = self.data_df.xs(symbol, level="symbol", drop_level=False)
            df = df.loc[(df.index.get_level_values("date") >= start_datetime.date()) &
                        (df.index.get_level_values("date") <= end_datetime.date())]
        except Exception as e:
            logger.warning(f"exception getting {symbol}")
            logger.warning(traceback.format_exc())
        df = df.reset_index()
        if df is None or df.empty:
            logger.info(f"why df is empty {df}")
        return df



    @staticmethod
    def add_vwap(df: pd.DataFrame):
        if "vwap" in df.columns.str.lower():
            logger.info("vwap already in dataframe, will not be added.")
            return df
        if "volume" not in df.columns:
            logger.info("volume NOT in dataframe, vwap will not be added.")
            return df
        if "amount" not in df.columns:
            logger.info("volume NOT in dataframe, vwap will not be added.")
            return df
        try:
            df["amount"] = df["amount"].replace('', np.nan)
            df["volume"] = df["volume"].replace('', np.nan)
            df = df.dropna(subset=["amount", "volume"])
            df["vwap"] = df["amount"].astype(float) / df["volume"].astype(float)
        except Exception as e:
            logger.error(f"compute vwap error amount {df['amount']}, volume {df['volume']}")
            logger.error(traceback.format_exc())
            df["vwap"] = np.nan
        return df


    def get_filtered_symbols(self) -> List[str]:
        rs = query_all_stock()
        data = pd.DataFrame(rs)
        name_pattern = r"退|ST|PT|指数"
        data = data[~data["简称"].str.contains(name_pattern, regex=True)]
        code_pattern = r"^(600|601|603|000)"
        data = data[data["代码"].str.contains(code_pattern, regex=True)]
        return data["代码"].sort_values().to_list()



    def get_instrument_list(self):
        logger.info("get filtered stock symbols......")
        symbols = self.get_filtered_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol: str):
        return str(symbol)

    def collector_data(self):
        """collector data"""
        self.download_index_data()
        self.data_df = self.download_db_data()
        super(DatabaseCollectorCN1d, self).collector_data()



class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=16, interval="1d", region="CN"):
        """
        Changed the default value of: scripts.data_collector.base.BaseRun.
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"DatabaseCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"DatabaseCollector{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """download data from Baostock

        Notes
        -----
            check_data_length, example:
                hs300 5min, a week: 4 * 60 * 5

        Examples
        ---------
            # get hs300 5min data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --start 2022-01-01 --end 2022-01-30 --interval 5min --region HS300
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Attention
        ---------
        qlib_data_1d_dir cannot be None, normalize 5min needs to use 1d data;

            qlib_data_1d can be obtained like this:
                $ python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn --version v3
            or:
                download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --normalize_dir ~/.qlib/stock_data/source/hs300_5min_nor --region HS300 --interval 5min
        """
        if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
            raise ValueError(
                "If normalize 5min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
            )

        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )




if __name__ == "__main__":
    fire.Fire(Run)


