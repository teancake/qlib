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



import qlib
from qlib.data import D

CUR_DIR = Path(__file__).resolve().parent
print(CUR_DIR)
sys.path.append(str(CUR_DIR.parent.parent))


from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import generate_minutes_calendar_from_daily, calc_adjusted_price, get_hs_stock_symbols

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"

class BaostockCollectorCN1d(BaseCollector):
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
        bs.login()
        super(BaostockCollectorCN1d, self).__init__(
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

    def download_index_data(self):
        # TODO: from MSN
        _format = "%Y%m%d"
        _begin = self.start_datetime.strftime(_format)
        _end = self.end_datetime.strftime(_format)
        for _index_name, _index_code in {"csi300": "000300", "csi100": "000903", "csi500": "000905"}.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                df = pd.DataFrame(
                    map(
                        lambda x: x.split(","),
                        requests.get(
                            INDEX_BENCH_URL.format(index_code=_index_code, begin=_begin, end=_end), timeout=None
                        ).json()["data"]["klines"],
                    )
                )
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            df.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.astype(float, errors="ignore")
            df["adjclose"] = df["close"]
            df["symbol"] = f"sh{_index_code}".upper()
            _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
            if _path.exists():
                _old_df = pd.read_csv(_path)
                df = pd.concat([_old_df, df], sort=False)
            df.to_csv(_path, index=False)
            time.sleep(5)



    def get_trade_calendar(self):
        _format = "%Y-%m-%d"
        start = self.start_datetime.strftime(_format)
        end = self.end_datetime.strftime(_format)
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        calendar_list = []
        while (rs.error_code == "0") & rs.next():
            calendar_list.append(rs.get_row_data())
        calendar_df = pd.DataFrame(calendar_list, columns=rs.fields)
        trade_calendar_df = calendar_df[~calendar_df["is_trading_day"].isin(["0"])]
        return trade_calendar_df["calendar_date"].values

    @staticmethod
    def process_interval(interval: str):
        return {"interval": "d", "fields": "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"}


    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = self.get_data_from_remote(
            symbol=symbol, interval=interval, start_datetime=start_datetime, end_datetime=end_datetime
        )
        df = df.rename(columns={"code": "symbol"})
        df["symbol"] = df["symbol"].map(lambda x: str(x).replace(".", "").upper())
        return df

    @staticmethod
    def get_data_from_remote(
        symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        logger.info(f"fetching {symbol} from {start_datetime} to {end_datetime}")
        rs = bs.query_history_k_data_plus(
            symbol,
            BaostockCollectorCN1d.process_interval(interval=interval)["fields"],
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency=BaostockCollectorCN1d.process_interval(interval=interval)["interval"],
            adjustflag="1",
        )
        if rs.error_code == "0" and len(rs.data) > 0:
            data_list = rs.data
            columns = rs.fields
            df = pd.DataFrame(data_list, columns=columns)
            df = BaostockCollectorCN1d.add_vwap(df)
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

        trade_dates = sorted(self.get_trade_calendar(), reverse=True)
        last_trade_date = datetime.today().date()
        for trade_date_str in trade_dates:
            trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d").date()
            if trade_date < last_trade_date:
                last_trade_date = trade_date
                break

        print("last trade date {}".format(last_trade_date))
        rs = bs.query_all_stock(last_trade_date)
        data = rs.get_data()
        name_pattern = r"退|ST|PT|指数"
        data = data[~data["code_name"].str.contains(name_pattern, regex=True)]
        code_pattern = r"\.(600|601|603|000)"
        data = data[data["code"].str.contains(code_pattern, regex=True)]
        return data["code"].sort_values().to_list()



    def get_hs300_symbols(self) -> List[str]:
        hs300_stocks = []
        trade_calendar = [self.get_trade_calendar().max()]
        print("trade calendar {}".format(trade_calendar))
        with tqdm(total=len(trade_calendar)) as p_bar:
            for date in trade_calendar:
                rs = bs.query_hs300_stocks(date=date)
                while rs.error_code == "0" and rs.next():
                    hs300_stocks.append(rs.get_row_data())
                p_bar.update()
        return sorted({e[1] for e in hs300_stocks})

    def get_instrument_list(self):
        logger.info("get filtered stock symbols......")
        symbols = self.get_filtered_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol: str):
        return str(symbol).replace(".", "").upper()

    def collector_data(self):
        """collector data"""
        self.download_index_data()
        super(BaostockCollectorCN1d, self).collector_data()



class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region="CN"):
        """
        Changed the default value of: scripts.data_collector.base.BaseRun.
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"BaostockCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"BaostockNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=10,
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

    def remove_index_instuments(self, instrments_dir: str = ""):
        # index sh.000, sz.399
        logger.info("remove indexes from instruments")
        instrments_dir = os.path.expanduser(instrments_dir)
        fn = os.path.join(instrments_dir, "all.txt")
        fn_out = os.path.join(instrments_dir, "filter.txt")
        with open(fn, "r") as fr, open(fn_out, "w") as fw:
            for line in fr:
                if line.split("\t")[0][:5] not in ["SH000", "SZ399"]:
                    fw.write(line)
        logger.info("done, filtered instrument file is {}".format(fn_out))




if __name__ == "__main__":
    fire.Fire(Run)


