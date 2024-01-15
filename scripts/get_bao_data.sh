#!/bin/zsh

# parameters
ds=$1
train_start=$2
pred_end=$3


# function definitions
function clean_up_old_dir() {
  ds=$1 # format "20231227"
  n_days_before=3
  for ((i = n_days_before; i < 10; i++)); do
    old_ds=$(date -d "$ds -$i days" "+%Y%m%d")
    stock_bao_cn_data=~/.qlib/stock_data/source/bao_cn_data_$old_ds
    qlib_bao_cn_data=~/.qlib/qlib_data/bao_cn_data_$old_ds
    echo "remove $stock_bao_cn_data"
    rm -r $stock_bao_cn_data
    echo "remove $qlib_bao_cn_data"
    rm -r $qlib_bao_cn_data
  done
}


# get bao data
echo "download bao data"
stock_bao_cn_data=~/.qlib/stock_data/source/bao_cn_data_$ds

if [[ -d $stock_bao_cn_data ]]; then
    echo "The directory $stock_bao_cn_data exists. Data won't be downloaded."
else
    python scripts/data_collector/baostock_1d/collector.py download_data --source_dir $stock_bao_cn_data --start $train_start --end $pred_end --interval 1d --region CN
fi


# dump bao data
qlib_bao_cn_data=~/.qlib/qlib_data/bao_cn_data_$ds

python scripts/dump_bin.py dump_all --csv_path $stock_bao_cn_data --qlib_dir $qlib_bao_cn_data --freq day --exclude_fields date,symbol

# generate new instrument file
python scripts/data_collector/baostock_1d/collector.py remove_index_instuments --instrments_dir=$qlib_bao_cn_data/instruments

# clean up old folders
echo "clean up data folders between "
clean_up_old_dir $ds

