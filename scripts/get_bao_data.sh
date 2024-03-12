#!/bin/zsh

# parameters
ds=$1
train_start=$2
pred_end=$3
stock_data_path_prefix=$4
qlib_data_path_prefix=$5

# function definitions
function clean_up_old_dir() {
  ds=$1 # format "20231227"
  n_days_before=3
  for ((i = n_days_before; i < 10; i++)); do
    old_ds=$(date -d "$ds -$i days" "+%Y%m%d")
    stock_cn_data=$stock_data_path_prefix_$old_ds
    qlib_cn_data=$qlib_data_path_prefix_$old_ds
    echo "remove $stock_cn_data"
    rm -r $stock_cn_data
    echo "remove $qlib_cn_data"
    rm -r $qlib_cn_data
  done
}


# get bao data
echo "download data"
stock_cn_data=$stock_data_path_prefix_$ds

if [[ -d $stock_cn_data ]]; then
    echo "The directory $stock_cn_data exists. Data won't be downloaded."
else
    python scripts/data_collector/database_1d/collector.py download_data --source_dir $stock_cn_data --start $train_start --end $pred_end --interval 1d --region CN
fi


# dump bao data
qlib_cn_data=$qlib_data_path_prefix_$ds

python scripts/dump_bin.py dump_all --csv_path $stock_cn_data --qlib_dir $qlib_cn_data --freq day --exclude_fields date,symbol

# generate new instrument file
python scripts/data_collector/database_1d/collector.py remove_index_instuments --instrments_dir=$qlib_cn_data/instruments

# clean up old folders
echo "clean up data folders between "
clean_up_old_dir $ds


