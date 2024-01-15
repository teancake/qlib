from sklearn.metrics import mean_squared_error
import numpy as np

from utils.log_util import get_logger
logger = get_logger()

from datetime import datetime

from utils.starrocks_db_util import StarrocksDbUtil

db_util = StarrocksDbUtil()


def compute_precision_recall(label, pred):
    print("#### label mse {}".format(mean_squared_error(label, pred)))

    auc_curve = []
    for th in range(-10, 10):
        label_th = th
        tp = len(label[np.logical_and(label >= label_th, pred >= th)])
        pp = len(pred[pred >= th])
        p = len(label[label >= th])
        n = len(label[label < th])
        fp = len(label[np.logical_and(label < label_th, pred >= th)])

        precision = tp / pp if pp > 0 else 0
        recall = tp / p if p > 0 else 0
        p_ratio = p/len(label)
        pp_ratio = pp / len(pred)
        tpr = recall
        fpr = fp / n if n > 0 else 0
        loss_p = len(label[np.logical_and(label < -label_th, pred >= th)]) / pp if pp > 0 else 0
        logger.info("threshold {}, precision {}, loss probability {}, recall {}, p ratio {}, pp ratio {}, pred len {}, label len {}, p cnt {}, pp cnt {}".format(th, precision, loss_p, recall, p_ratio,
                                                                                      pp_ratio, len(pred), len(label), p, pp))
        auc_curve.append([th, precision, recall, tpr, fpr, pp_ratio])
    auc_curve = np.array(auc_curve)
    # print(auc_curve)


def save_prediction_to_db(data_ext, ds, model_name, stage, pred_name, pred_val_col_name, label_col_name, run_id):
    df = data_ext.loc[:, ["日期","代码", pred_val_col_name, label_col_name]]
    table_name = "ods_stock_zh_a_prediction"
    logger.info(
        "saving {} results to db, table name {}, ds {}, model {}, stage {}, run_id {}".format(stage, table_name, ds,
                                                                                            model_name,
                                                                                            stage, run_id))
    df["ds"] = ds
    df["gmt_create"] = datetime.now()
    df["gmt_modified"] = datetime.now()
    df["stage"] = stage
    df["model"] = model_name
    df["run_id"] = run_id
    df["pred_name"] = pred_name
    df["pred_value"] = df[pred_val_col_name]
    if pred_val_col_name != "pred_value":
        df = df.drop(columns=[pred_val_col_name])
    df["label"] = df[label_col_name]
    if label_col_name != "label":
        df = df.drop(columns=[label_col_name])

    df.to_sql(name=table_name, con=db_util.get_db_engine(), if_exists="append", index=False, method="multi",
              chunksize=1000)
    logger.info("{} results saved to db.".format(stage))
