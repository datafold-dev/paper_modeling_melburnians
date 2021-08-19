#!/usr/bin/env python3

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
from datafold.appfold import EDMD
from datafold.appfold.edmd import EDMDWindowPrediction
from datafold.dynfold import (
    DiffusionMaps,
    LocalRegressionSelection,
    TSCTakensEmbedding,
)

from datafold.dynfold.dmd import DMDFull
from datafold.pcfold import (
    GaussianKernel,
    TSCDataFrame,
    TSCMetric,
)

from datafold.utils.plot import (
    plot_eigenvalues_time,
)
from matplotlib import rc
from pandas.api.types import is_timedelta64_dtype
from sklearn.model_selection import train_test_split

rc("text", usetex=True)
pylab.rcParams[
    "text.latex.preamble"
] = r"\usepackage{tgheros} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}"
rc("font", size=12)

SINGLE_COLUMN = 8  # inches
ONEHALF_COLUMN = 10  # inches
DOUBLE_COLUMN = 14  # inches


class EDMDPositiveSensors(EDMD):
    def _predict_ic(self, X_dict: TSCDataFrame, time_values, qois) -> TSCDataFrame:
        _ret = super(EDMDPositiveSensors, self)._predict_ic(X_dict, time_values, qois)
        _ret_sensors = _ret.loc[:, _ret.columns.str.startswith("sensor_")].copy()
        _ret_sensors[_ret_sensors < 0] = 0
        _ret.loc[:, _ret.columns.str.startswith("sensor_")] = _ret_sensors
        return _ret


class EpsGaussianKernel(object):
    def __init__(self, factor):
        self.factor = float(factor)

    def __call__(self, d):
        return np.median(d) * self.factor

    def __repr__(self):
        return f"median(Distance)*{self.factor}"


def print_data_info(X):
    print("======================================================")
    print(f"X.shape={X.shape}")
    print("----------------------------")
    print(f"melbourne_data.n_timeseries={X.n_timeseries}")
    print("----------------------------")
    print(f"melbourne_data.n_samples={X.shape[0]}")
    print("----------------------------")
    print(f"melbourne_data.n_timesteps={X.n_timesteps}")
    print("----------------------------")
    print(f"melbourne_data.n_sensors={X.shape[1]}")
    print("----------------------------")
    print(f"check Null values: {X.isnull().sum(axis=0)}")
    print("======================================================")


def filter_data(X, start_time, n_samples_ic, min_timesteps):
    assert start_time < 24

    if start_time is not None:
        # start bocks such that the prediction starts at the same specified time
        align_ts = list()
        for _ts_id, _ts in X.itertimeseries():

            if isinstance(_ts.index, pd.DatetimeIndex):
                hour_per_day = _ts.index.hour
            else:
                hour_per_day = np.array(np.mod(_ts.index, 24))
            ts_start_time = start_time - n_samples_ic + 1

            if ts_start_time < 0:
                ts_start_time = np.mod(24 - np.mod(ts_start_time, 24), 24)

            try:
                start_idx = np.argwhere(hour_per_day == ts_start_time)[0][0]
                align_ts.append(_ts.iloc[start_idx:, :])
            except Exception as e:
                if _ts.shape[0] < 24 and ts_start_time not in hour_per_day:
                    # it can happen that there are only very short time series,
                    # these are dropped here already
                    pass
                else:
                    raise e

        X = TSCDataFrame.from_frame_list(align_ts)

    n_timesteps = X.n_timesteps

    if isinstance(X.n_timesteps, pd.Series):
        drop_ids_timesteps = n_timesteps[X.n_timesteps < min_timesteps].index
        if not drop_ids_timesteps.empty:
            X = X.drop(drop_ids_timesteps, level=0)
            X = X.tsc.assign_ids_sequential()

    return X


def time_interval_and_sensor_selection(X):
    # considered for publication, long time horizon, even number of sensors
    X = X.loc[
        :,
        (
            # "sensor_1_counts", #i:2009/03/24
            "sensor_2_counts",  # i:2009/03/30
            # "sensor_3_counts", #i:2009/03/25
            # "sensor_4_counts", #i:2009/03/23
            # "sensor_5_counts", #i:2009/03/26
            "sensor_6_counts",  # i:2009/03/25
            # "sensor_7_counts", #i:2014/12/17
            # "sensor_8_counts",  # i:2009/03/24
            "sensor_9_counts",  # i:2009/03/23
            "sensor_10_counts",  # i:2009/04/23
            # "sensor_11_counts", #i:2009/01/20
            # "sensor_12_counts", #i:2009/01/21
            # "sensor_13_counts", #i:2009/03/24  # no
            # "sensor_14_counts", #i:2019/09/25
            # "sensor_15_counts", #i:2009/03/25
            # "sensor_16_counts", #i:2009/03/30 || n:Device moved to location ID 53 (22/09/2015)
            # "sensor_17_counts", #i:2009/03/30 || n:Device is upgraded in 26/02/2020
            "sensor_18_counts",  # i:2009/03/30
            # "sensor_19_counts",  # i:2013/09/02
            # "sensor_20_counts", #i:2013/09/06
            "sensor_21_counts",  # i:2013/09/02
            # "sensor_22_counts", #i:2013/08/12
            # "sensor_23_counts", #i:2013/09/02
            "sensor_24_counts",  # i:2013/09/02
            # "sensor_25_counts",  # i:2019/10/02 || n:Sensor relocated from 14 to 25 on 2/10/2019
            "sensor_26_counts",  # i:2013/09/28
            "sensor_27_counts",  # i:2013/08/16
            "sensor_28_counts",  # i:2013/08/23
            # "sensor_29_counts", #i:2013/10/11 || n:sensor upgraded from laser to 3D on 19/12/2019
            # "sensor_30_counts", #i:2013/10/14
            "sensor_31_counts",  # i:2013/10/10
            # "sensor_32_counts", #i:2013/12/20 || n:Device has been removed (24/01/2017)
            # "sensor_33_counts", #i:2014/04/23
            # "sensor_34_counts", #i:2014/06/08
            # "sensor_35_counts", #i:2016/04/11
            # "sensor_36_counts", #i:2015/01/20
            # "sensor_37_counts", #i:2015/02/11
            # "sensor_38_counts", #i:2014/12/05 || n:Device has been removed (17/02/2017)
            # "sensor_39_counts", #i:2019/12/04 || n:In 4/12/2019 sensor upgraded
            # "sensor_40_counts", #i:2015/01/19
            # "sensor_41_counts", #i:2017/06/29
            # "sensor_42_counts", #i:2015/04/15
            # "sensor_43_counts", #i:2015/04/15
            # "sensor_44_counts", #i:2015/04/15
            # "sensor_45_counts", #i:2017/06/29
            # "sensor_46_counts", #i:2017/07/10
            # "sensor_47_counts", #i:2017/08/24
            # "sensor_48_counts", #i:2017/10/02
            # "sensor_49_counts", #i:2017/11/29
            # "sensor_50_counts", #i:2017/11/30
            # "sensor_51_counts", #i:2017/11/30
            # "sensor_52_counts", #i:2017/07/31
            # "sensor_53_counts", #i:2015/09/23
            # "sensor_54_counts", #i:2018/06/26
            # "sensor_55_counts", #i:2018/07/19
            # "sensor_56_counts", #i:2018/07/25
            # "sensor_57_counts", #i:2018/08/13
            # "sensor_58_counts", #i:2018/09/27
            # "sensor_59_counts", #i:2019/02/13
            # "sensor_60_counts", #i:2019/03/08 || n:Temporary for the duration of the metro tunnel works. Installed under the scaffolding.
            # "sensor_61_counts", #i:2019/06/28
            # "sensor_62_counts", #i:2019/09/25
            # "sensor_63_counts", #i:2020/01/07
            # "sensor_64_counts", #i:2020/01/16
            # "sensor_65_counts", #i:2020/03/12
            # "sensor_66_counts", #i:2020/04/06
            # "sensor_67_counts", #i:2020/06/03
            # "sensor_68_counts", #i:2020/06/03
            # "sensor_69_counts", #i:2020/06/03
            # "sensor_70_counts", #i:2020/10/12
            # "sensor_71_counts", #i:2020/10/16
            # "sensor_72_counts", #i:2020/11/30
            # "sensor_73_counts", #i:2020/10/02
            # "sensor_75_counts", #i:2020/12/18
        ),
    ]

    start_date = np.datetime64("2016-01-01T00:00:00")
    end_date = np.datetime64("2020-01-01T00:00:00")
    all_dates = X.index.get_level_values("time")

    mask = (all_dates >= start_date) & (all_dates < end_date)
    X = X.loc[mask]
    return X


def read_and_select_data(filename):
    melbourne_data: TSCDataFrame = TSCDataFrame.from_csv(filename, parse_dates=True)
    melbourne_data = time_interval_and_sensor_selection(X=melbourne_data)

    # Remove holidays on weekdays:
    mask_holidays = np.array(
        list(
            map(
                lambda x: x in holidays.Australia(prov="VIC"),
                melbourne_data.time_values().astype(str),
            )
        )
    )
    weekdays = pd.DatetimeIndex(melbourne_data.time_values()).dayofweek < 5
    mask_holidays = np.logical_and(mask_holidays, weekdays)
    print(f"Remove {mask_holidays.sum()} hours from holidays.")
    melbourne_data = melbourne_data.loc[~mask_holidays]

    n_samples = melbourne_data.shape[0]
    tsc_id_idx = pd.MultiIndex.from_arrays(
        [np.zeros(n_samples), melbourne_data.index.get_level_values("time")]
    )

    melbourne_data = TSCDataFrame.from_same_indices_as(
        indices_from=melbourne_data, values=melbourne_data, except_index=tsc_id_idx
    )

    for col in melbourne_data.columns:
        sensor_column = melbourne_data.loc[:, col]
        sensor_column = sensor_column.loc[sensor_column.isnull().sum(axis=1) == 0, :]
        assigned = sensor_column.tsc.assign_ids_const_delta(drop_samples=True)
        print(f"{col} has {assigned.n_timeseries} timeseries")
        print(f"{col} has {assigned.delta_time} delta_time")

    # melbourne_data = melbourne_data.groupby("ID").fillna(
    #     method="ffill", axis=0, limit=2
    # )

    melbourne_data = melbourne_data.tsc.assign_ids_const_delta(drop_samples=False)

    melbourne_data = melbourne_data.loc[melbourne_data.isnull().sum(axis=1) == 0, :]
    melbourne_data = melbourne_data.tsc.assign_ids_const_delta(drop_samples=True)
    assert not melbourne_data.isnull().any().any()

    delta_times = melbourne_data.delta_time
    if isinstance(delta_times, pd.Series):
        if is_timedelta64_dtype(delta_times):
            _cmp = np.timedelta64(1, "h")
        else:
            _cmp = 1

        drop_ids_delta_time = delta_times[delta_times != _cmp].index
        melbourne_data = melbourne_data.drop(drop_ids_delta_time, level=0)
        melbourne_data = melbourne_data.tsc.assign_ids_sequential()

    print_data_info(melbourne_data)

    return melbourne_data


def setup_basic_edmd():

    takens_delays = 168

    takens = (
        "takens",
        TSCTakensEmbedding(
            delays=takens_delays,
            kappa=0,  # kappa values larger than 0 did not show improvement
        ),
    )

    laplace = (
        "laplace",
        DiffusionMaps(
            kernel=GaussianKernel(epsilon=EpsGaussianKernel(factor=1)),
            n_eigenpairs=500,  # 500 optimal in one CV
            alpha=1,
            time_exponent=0,  # -0.5 is a scaling factor by Giannakis for the DMAP
            symmetrize_kernel=True,
            dist_kwargs=dict(cut_off=np.inf),  # exact_numeric=False, n_jobs=-1
        ),
    )

    edmd = EDMDPositiveSensors(
        dict_steps=[takens, laplace],
        dmd_model=DMDFull(is_diagonalize=True),
        include_id_state=False,
        sort_koopman_triplets=True,
        verbose=True,
    )

    prediction_window = int(24)
    n_samples_ic = takens_delays + 1
    window_size = prediction_window + n_samples_ic

    edmd = EDMDWindowPrediction(
        window_size=window_size, offset=prediction_window
    ).adapt_model(edmd)

    print("Model: \n", edmd)
    return edmd, n_samples_ic


def plot_sensor_time_series(
    X_original,
    X_reconstruct_train,
    diff_train,
    initial_states_train,
    X_reconstruct_test,
    diff_test,
    initial_states_test,
):

    sensor_columns = X_original.columns[X_original.columns.str.startswith("sensor_")]
    sensor_columns = ["sensor_2_counts", "sensor_27_counts"]

    for _col_sensor in sensor_columns:

        f, ax = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw=dict(height_ratios=[5, 1]),
            figsize=[15, 9],
        )

        _df = X_original.loc[:, _col_sensor]

        _df_pred_train = X_reconstruct_train.loc[:, _col_sensor]
        _df_diff_train = diff_train.loc[:, _col_sensor]
        _df_initial_states_train = initial_states_train.loc[:, [_col_sensor]]

        _df_pred_test = X_reconstruct_test.loc[:, _col_sensor]
        _df_diff_test = diff_test.loc[:, _col_sensor]
        _df_initial_states_test = initial_states_test.loc[:, [_col_sensor]]

        ax[0].set_title(_col_sensor)
        ax[0].grid()

        for _ts_id, _ts in TSCDataFrame(_df).itertimeseries():
            ax[0].plot(_ts.index, _ts.to_numpy(), c="black")

        _df_pred_train.plot(ax=ax[0], legend=False, color="green")

        ax[0].plot(
            _df_initial_states_train.time_values(),
            _df_initial_states_train.to_numpy().ravel(),
            "o",
            markersize=5,
            c="green",
        )

        _df_pred_test.plot(ax=ax[0], legend=False, color="red")
        ax[0].plot(
            _df_initial_states_test.time_values(),
            _df_initial_states_test.to_numpy().ravel(),
            "o",
            markersize=5,
            c="red",
        )

        index = np.hstack(
            [
                _df_pred_train.time_values_delta_time(),
                _df_pred_test.time_values_delta_time(),
            ]
        )

        _tmp_train = pd.DataFrame(_df_pred_train).droplevel(0)
        _tmp_test = pd.DataFrame(_df_pred_test).droplevel(0)

        upper_limit = (
            max(
                [
                    float(_df_pred_train.max()),
                    float(_df_pred_test.max()),
                    float(_df.max(axis=0)),
                ]
            )
            * 1.1
        )

        ax[0].set_ylim([-50, upper_limit])


def plot_paper_data(train_sensor_data, test_sensor_data, n_samples_ic):
    sensors = train_sensor_data.columns

    f, ax = plt.subplots(
        nrows=len(sensors),
        sharex=True,
        sharey=True,
        figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.85),
    )
    f.subplots_adjust(hspace=0, left=0.207, right=0.917, top=0.94, bottom=0.19)
    for i, col in enumerate(sensors):
        _train = train_sensor_data.loc[:, [col]]
        for j, ts in _train.itertimeseries():
            ts_warm_up = ts.iloc[:n_samples_ic, :]
            ts_data = ts.iloc[n_samples_ic:, :]

            ax[i].fill_between(
                ts_warm_up.index, y1=ts_warm_up.to_numpy().ravel(), y2=0, color="black"
            )
            ax[i].fill_between(
                ts_data.index, y1=ts_data.to_numpy().ravel(), y2=0, color="skyblue"
            )

        _test = test_sensor_data.loc[:, col]
        for j, ts in _test.itertimeseries():
            ts_warm_up = ts.iloc[:n_samples_ic, :]
            ts_data = ts.iloc[n_samples_ic:, :]

            ax[i].fill_between(
                ts_warm_up.index, y1=ts_warm_up.to_numpy().ravel(), y2=0, color="black"
            )
            ax[i].fill_between(
                ts_data.index, y1=ts_data.to_numpy().ravel(), y2=0, color="lightcoral"
            )

        ax[i].axes.set_yticklabels([])
        ax[i].axes.set_yticks([0, 7000])
        ax[i].set_ylim([0, 7000])
        ax[i].set_ylabel(col.replace("sensor_", "").replace("_counts", ""), rotation=0)

        ax[i].set_xlim([np.datetime64("2016-01-01"), np.datetime64("2019-12-31")])
        ax[i].axes.set_xticks(
            [
                np.datetime64("2016-01-01"),  # start training
                np.datetime64("2017-01-01"),  # start training
                # np.datetime64("2018-08-06"),  # end training
                np.datetime64("2018-09-01"),  # start test
                np.datetime64("2019-12-31"),  # end test
            ]
        )
        ax[i].axes.set_xticklabels(["2016.1", "2017.1", "2018.9", "2020.1"])

    f.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("time [year.month]")
    plt.ylabel("sensor ID (s)")


def plot_error_table(
    edmd,
    X_windows_train: TSCDataFrame,
    X_reconstruct_train: TSCDataFrame,
    X_windows_test: TSCDataFrame,
    X_reconstruct_test: TSCDataFrame,
):
    def predict_naive(X_windows, X_reconstruct):
        X_reconstruct_naive = []

        for i, df in X_reconstruct.itertimeseries():
            last_week_index = df.index - np.timedelta64(7, "D")

            naive_df = X_windows.loc[pd.IndexSlice[i, last_week_index], :]
            naive_df.index = pd.MultiIndex.from_product([[i], df.index.copy()])
            X_reconstruct_naive.append(naive_df)

        X_reconstruct_naive = TSCDataFrame.from_frame_list(X_reconstruct_naive)
        return X_reconstruct_naive

    X_reconstruct_train_naive = predict_naive(
        X_windows=X_windows_train, X_reconstruct=X_reconstruct_train
    )
    X_reconstruct_test_naive = predict_naive(
        X_windows=X_windows_test, X_reconstruct=X_reconstruct_test
    )

    # COMPUTE MER
    mer_true = pd.DataFrame(X_windows_test.loc[X_reconstruct_test.index, :]).copy()
    mer_pred = pd.DataFrame(X_reconstruct_test).copy()

    mer_true = mer_true.drop(mer_true.groupby("ID").head(1).index, axis=0)
    mer_pred = mer_pred.drop(mer_pred.groupby("ID").head(1).index, axis=0)

    nominator = (mer_true - mer_pred).abs()

    denom = mer_true.groupby("ID").mean()

    nominator = pd.DataFrame(nominator.drop(55, axis=0, level=0))
    denom = pd.DataFrame(denom.drop(55, axis=0))

    mer = (nominator / denom).groupby("ID").sum()
    mer = (100 * 1 / 24 * mer).mean(axis=0)
    mer = np.mean(mer)

    print(f"MER on entire test set {mer}")

    edmd.metric_eval = TSCMetric(metric="mae", mode="feature", scaling="min-max")

    edmd_score_train = edmd._score_eval(
        X_windows_train.loc[X_reconstruct_train.index, :], X_reconstruct_train
    )
    naive_score_train = edmd._score_eval(
        X_windows_train.loc[X_reconstruct_train.index, :], X_reconstruct_train_naive
    )

    edmd_score_test = edmd._score_eval(
        X_windows_test.loc[X_reconstruct_test.index, :], X_reconstruct_test
    )
    naive_score_test = edmd._score_eval(
        X_windows_test.loc[X_reconstruct_test.index, :], X_reconstruct_test_naive
    )

    print(f"EDMD score train = {edmd_score_train}")
    print(f"Naive score train = {naive_score_train}")

    print(f"EDMD score test = {edmd_score_test}")
    print(f"Naive score test = {naive_score_test}")

    y_true_train = X_windows_train.loc[X_reconstruct_train.index]
    y_pred_train = X_reconstruct_train

    y_true_test = X_windows_test.loc[X_reconstruct_test.index]
    y_pred_test = X_reconstruct_test

    idx = [
        i.replace("_counts", "").replace("sensor_", "") for i in X_windows_test.columns
    ]

    scale = scale_sensors(X_windows=X_windows_train.loc[X_reconstruct_train.index, :])

    error_table = pd.DataFrame(
        scale.astype(int).astype(str).to_numpy(),
        index=idx,
        columns=pd.MultiIndex.from_arrays([[""], ["$Q_{95\%}$"]]),
    )

    col = pd.MultiIndex.from_product(
        [["training", "test"], ["mean $\pm$ std", "mean", "std", "RRMSE", "RRMSE(b)"]]
    )
    error_table = pd.concat(
        [error_table, pd.DataFrame("", index=idx, columns=col)], axis=1
    )

    error_table.loc[:, ("training", "mean")] = (
        (y_true_train - y_pred_train).mean().to_numpy()
    )
    error_table.loc[:, ("training", "std")] = (
        (y_true_train - y_pred_train).std().to_numpy()
    )

    error_table.loc[:, ("test", "mean")] = (y_true_test - y_pred_test).mean().to_numpy()
    error_table.loc[:, ("test", "std")] = (y_true_test - y_pred_test).std().to_numpy()

    error_metric = TSCMetric(metric="rmse", mode="feature")

    train_error_edmd = (
        error_metric(
            y_true=y_true_train,
            y_pred=y_pred_train,
        )
        / scale
        * 100
    ).to_numpy()

    train_error_naive = (
        error_metric(
            X_windows_train.loc[X_reconstruct_train_naive.index, :],
            X_reconstruct_train_naive,
        )
        / scale
        * 100
    ).to_numpy()

    test_error_edmd = (
        error_metric(y_true=y_true_test, y_pred=y_pred_test).to_numpy() / scale * 100
    ).to_numpy()

    test_error_naive = (
        error_metric(
            X_windows_test.loc[X_reconstruct_test.index, :], X_reconstruct_test_naive
        )
        / scale
        * 100
    ).to_numpy()

    from numpy.core.defchararray import add as npaddstr

    error_table.loc[:, ("training", "RRMSE")] = np.round(train_error_edmd, 2).astype(
        str
    )
    error_table.loc[:, ("test", "RRMSE")] = np.round(test_error_edmd, 2).astype(str)

    error_table.loc[:, ("training", "RRMSE(b)")] = np.round(
        train_error_naive, 2
    ).astype(str)

    error_table.loc[:, ("test", "RRMSE(b)")] = np.round(test_error_naive, 2).astype(str)

    avr = lambda vals: np.round(np.average(vals), 2).astype(str)
    weight_avr = lambda vals: np.average(vals, weights=scale)

    error_table.loc["agg.", :] = [
        "",  # scale
        "",  # mean pm std
        weight_avr(error_table.loc[:, ("training", "mean")]),
        np.sqrt(
            weight_avr(np.square(error_table.loc[:, ("training", "std")].to_numpy()))
        ),
        avr(error_table.loc[:, ("training", "RRMSE")].astype(float)),
        avr(error_table.loc[:, ("training", "RRMSE(b)")].astype(float)),
        "",  # mean pm std
        np.round(weight_avr(error_table.loc[:, ("test", "mean")]), 2),
        np.sqrt(weight_avr(np.square(error_table.loc[:, ("test", "std")].to_numpy()))),
        avr(error_table.loc[:, ("test", "RRMSE")].astype(float)),
        avr(error_table.loc[:, ("test", "RRMSE(b)")].astype(float)),
    ]

    def mean_pm_std(_mean, _std):
        return [
            f"{np.round(i).astype(int)} $\pm$ {np.round(j).astype(int)}"
            for i, j in zip(
                _mean,
                _std,
            )
        ]

    error_table.loc[:, ("training", "mean $\pm$ std")] = mean_pm_std(
        error_table.loc[:, ("training", "mean")].to_numpy(),
        error_table.loc[:, ("training", "std")].to_numpy(),
    )

    error_table.loc[:, ("test", "mean $\pm$ std")] = mean_pm_std(
        error_table.loc[:, ("test", "mean")].to_numpy(),
        error_table.loc[:, ("test", "std")].to_numpy(),
    )

    error_table = error_table.drop(error_table.columns[[2, 3, 7, 8]], axis=1)

    error_table.loc[:, error_table.columns[[2, 3, 5, 6]]] = npaddstr(
        error_table.loc[:, error_table.columns[[2, 3, 5, 6]]].to_numpy().astype(str),
        " \%",
    )

    error_table.columns = error_table.columns.set_levels(
        [
            "",
            f"test ($C_{{\\text{{test}}}} = {X_reconstruct_test.n_timeseries}$)",
            f"training ($C_{{\\text{{train}}}} = {X_reconstruct_train.n_timeseries}$)",
        ],
        level=0,
    )

    error_table.index.name = "ID ($s$)"
    error_table = error_table.reset_index(col_level=1)

    return error_table


def plot_paper_week_timeseries(X_windows_test, X_reconstruct_test):

    ic_time = pd.DatetimeIndex(
        X_windows_test.initial_states().index.get_level_values("time")
    )

    mondays_idx = np.where(ic_time.dayofweek == 0)[0]  # 0 = Monday, 6 = Sunday
    mondays_idx = mondays_idx[mondays_idx + 14 < len(ic_time)]

    timedelta = ic_time[mondays_idx + 14] - ic_time[mondays_idx]

    X_windows_test = X_windows_test.loc[X_reconstruct_test.index, :]

    sensor_selection = [f"sensor_{s}_counts" for s in [2, 9, 28, 31]]
    fortnights_ids = [113]  # 215

    for plot_id in fortnights_ids:

        plot_data_test = X_windows_test.loc[plot_id : plot_id + 6, :]
        plot_data_reconstruct = X_reconstruct_test.loc[plot_id : plot_id + 6, :]

        f, ax = plt.subplots(
            nrows=len(sensor_selection),
            sharex=True,
            figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.55),
        )
        f.subplots_adjust(bottom=0.136, top=0.893)

        for i, sensor in enumerate(sensor_selection):
            sensor_data_test = plot_data_test.loc[:, sensor]
            sensor_data_reconstruct = plot_data_reconstruct.loc[:, sensor]

            is_last = i == len(sensor_selection) - 1

            ics = sensor_data_reconstruct.initial_states()

            ax[i].plot(
                ics.index.get_level_values("time"),
                ics.to_numpy().ravel(),
                "o",
                markersize=5,
                color="mediumblue",
                label="initial condition" if is_last else None,
            )

            ax[i].plot(
                sensor_data_test.index.get_level_values("time"),
                sensor_data_test.to_numpy().ravel(),
                color="black",
                label="true values" if is_last else None,
            )

            ax[i].plot(
                sensor_data_reconstruct.index.get_level_values("time"),
                sensor_data_reconstruct.to_numpy(),
                color="red",
                alpha=0.7,
                label="predicted values" if is_last else None,
            )
            ax[i].set_xticks(
                np.arange(
                    np.datetime64("2019-02-11"),
                    np.datetime64("2019-02-19"),
                    np.timedelta64(1, "D"),
                )
            )
            ax[i].set_xticklabels([f"{i}.02." for i in range(11, 19)])
            ax[i].set_xlim([np.datetime64("2019-02-11"), np.datetime64("2019-02-18")])

            max_val = np.max(sensor_data_test.to_numpy().ravel())
            ytick = np.round(max_val, decimals=-2)
            ytick = int(ytick)
            ax[i].set_yticks([max_val // 2, ytick])
            ax[i].set_yticklabels(["", ""])
            ax[i].set_ylim([0, ytick])

            ax_right = ax[i].twinx()
            ax_right.set_ylabel(
                sensor.replace("sensor_", "").replace("_counts", ""),
                labelpad=10,
                rotation=0,
            )
            ax_right.set_yticks([])

            ax[i].set_ylabel(ytick, labelpad=10, rotation=0)
            ax[i].grid()
        f.align_ylabels()

        if is_last:
            f.legend(loc="upper center", ncol=3)

    ax = f.add_subplot(111, frame_on=False)
    ax1 = ax.twinx()
    ax.tick_params(labelcolor="none", bottom=False, left=False, right=False)
    ax.set_xlabel("time (year 2019)")
    ax.set_ylabel("pedestrian count", labelpad=10)

    ax1.tick_params(labelcolor="none", bottom=False, left=False, right=False)
    ax1.set_ylabel("sensor ID", labelpad=5, rotation=-90)


def scale_sensors(X_windows):
    quantile95 = pd.DataFrame(X_windows).quantile(q=0.95)
    return quantile95


def plot_paper_sensor_profile(
    scale,
    X_windows_test,
    X_reconstruct_test,
):
    sensor_columns = X_reconstruct_test.columns[
        X_reconstruct_test.columns.str.startswith("sensor_")
    ]

    X_reconstruct_test = X_reconstruct_test.loc[:, sensor_columns]

    X_reconstruct_test = X_reconstruct_test.drop(
        X_reconstruct_test.groupby("ID").head(1).index, axis=0
    )
    X_windows_test = X_windows_test.drop(
        X_windows_test.groupby("ID").head(1).index, axis=0
    )

    X_win_scaled = X_windows_test.loc[X_reconstruct_test.index, :] / scale
    X_reconstruct_scaled = X_reconstruct_test / scale

    sensor_columns = X_windows_test.columns[
        X_windows_test.columns.str.startswith("sensor_")
    ]

    f, ax = plt.subplots(
        nrows=3,
        ncols=len(sensor_columns) + 3,
        figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.5),
        gridspec_kw={
            "width_ratios": np.append(np.ones(len(sensor_columns) + 1), [0.3, 0.2])
        },
    )

    gs = ax[0, -1].get_gridspec()

    # from https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
    # remove the underlying axes
    for _ax in ax[0:2, -1].ravel():
        _ax.remove()
    axcbar_vals = f.add_subplot(gs[0:2, -1])

    # f.suptitle(column)
    f.subplots_adjust(
        hspace=0.14, wspace=0.02, bottom=0.138, top=0.921, left=0.088, right=0.943
    )
    # left=0.048, bottom=0.1, right=0.9, # top=0.974,

    ax[0][-2].remove()
    ax[1][-2].remove()
    ax[2][-2].remove()
    # ax[1][-1].remove()

    cmap = "GnBu"

    for i, column in enumerate(sensor_columns):

        is_first = i == 0
        is_last = i == len(sensor_columns) - 1

        true_values = X_win_scaled.loc[:, column]
        pred_values = X_reconstruct_scaled.loc[:, column]

        true_values = true_values.to_numpy().reshape(
            [true_values.n_timeseries, true_values.n_timesteps]
        )

        pred_values = pred_values.to_numpy().reshape(
            [pred_values.n_timeseries, pred_values.n_timesteps]
        )

        vmin, vmax = 0, 1

        vals = ax[0][i].imshow(
            true_values, vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap
        )
        ax[0][i].set_title(
            column.replace("sensor_", "").replace("_counts", ""),
            # + f"\n({int(mean_max_val_per_day[i]):d})"
        )
        ax[1][i].imshow(pred_values, vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap)

        vmin, vmax = -0.5, 0.5
        ax[2][i].imshow(
            pred_values - true_values, aspect="auto", vmin=vmin, vmax=vmax, cmap="bwr"
        )
        # ax[2][i].set_xlabel(
        #     f"{int(mean_absolute_error(X_windows_test.loc[:, column], X_reconstruct_test.loc[:, column]))}"
        # )

        if is_first:
            ax[0][0].set_xticks([])
            ax[1][0].set_xticks([])
            ax[2][0].set_xticks([0, 23])
            ax[2][0].set_xticklabels([1, 24])
            ax[0][0].set_yticks([0, 100, 200, 300])
            ax[1][0].set_yticks([0, 100, 200, 300])
            ax[2][0].set_yticks([0, 100, 200, 300])
        else:
            ax[0][i].set_xticks([])
            ax[1][i].set_xticks([])
            ax[2][i].set_xticks([23])
            ax[2][i].set_xticklabels([24])
            ax[0][i].set_yticks([])
            ax[1][i].set_yticks([])
            ax[2][i].set_yticks([])

        if is_last:
            cbar = plt.colorbar(vals, axcbar_vals, ticks=[0, 0.5, 1], extend="max")
            cbar.ax.set_yticklabels(["0", "0.5", "1"])

    mean_true_values = TSCDataFrame(
        np.average(X_win_scaled.values, axis=1),
        index=X_win_scaled.index,
        columns=["average"],
    )
    mean_true_values = mean_true_values.to_numpy().reshape(
        [mean_true_values.n_timeseries, mean_true_values.n_timesteps]
    )

    mean_pred_values = TSCDataFrame(
        np.average(
            X_reconstruct_scaled.values, axis=1
        ),  # , weights=mean_max_val_per_day
        index=X_win_scaled.index,
        columns=["average"],
    )
    mean_pred_values = mean_pred_values.to_numpy().reshape(
        [mean_pred_values.n_timeseries, mean_pred_values.n_timesteps]
    )

    c = -3
    ax[0][c].imshow(mean_true_values, vmin=0, vmax=1, aspect="auto", cmap=cmap)
    ax[1][c].imshow(mean_pred_values, vmin=0, vmax=1, aspect="auto", cmap=cmap)

    err = ax[2][c].imshow(
        mean_pred_values - mean_true_values,
        vmin=-0.5,
        vmax=0.5,
        cmap="bwr",
        aspect="auto",
    )

    cbar = plt.colorbar(err, ax[2][-1], ticks=[-0.5, 0, 0.5], extend="both")
    cbar.ax.set_yticklabels([-0.5, 0, 0.5])

    ax[0][c].set_title("agg.")

    ax[0][c].set_xticks([])
    ax[1][c].set_xticks([])
    ax[2][c].set_xticks([23])
    ax[2][c].set_xticklabels([24])
    ax[0][c].set_yticks([])
    ax[1][c].set_yticks([])
    ax[2][c].set_yticks([])

    ax[0][c].set_ylabel("true values", labelpad=10, rotation=-90)
    ax[1][c].set_ylabel("pred. values", labelpad=10, rotation=-90)
    ax[2][c].set_ylabel("difference", labelpad=10, rotation=-90)

    ax[0][c].yaxis.set_label_position("right")
    ax[1][c].yaxis.set_label_position("right")
    ax[2][c].yaxis.set_label_position("right")

    f.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("time [hour]")
    plt.ylabel("initial condition")


def plot_timeseries_dmap(
    sensor_str, dmap_idx, X_windows_train, X_windows_test, X_latent_train, X_latent_test
):

    f, ax = plt.subplots(
        nrows=2, sharex=True, figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.55)
    )

    X_windows_train.loc[:, sensor_str].plot(
        ax=ax[0], legend=False, ylabel="", c="black"
    )
    X_windows_test.loc[:, sensor_str].plot(
        ax=ax[0], ylabel="", legend=False, c="black"
    )

    # -1 bc. in paper is 1-indexed, in code 0-indexed
    X_latent_train.loc[:, f"dmap{dmap_idx-1}"].plot(ax=ax[1], legend=False, c="black")
    X_latent_test.loc[:, f"dmap{dmap_idx-1}"].plot(ax=ax[1], legend=False, c="black")

    ax[0].set_ylabel(sensor_str.replace("_", ""))
    ax[1].set_ylabel(fr"$\varphi_{{{dmap_idx}}}$")


def plot_paper_dmap_selection(
    edmd,
    selection,
    X_latent_test,
):

    dmap_eigenvalues = edmd.named_steps["laplace"].eigenvalues_
    print(f"Smallest DMAP eigenvalue = {dmap_eigenvalues[-1]}")

    sensor_mask = X_latent_test.columns.str.startswith("sensor_")
    X_latent_test = X_latent_test.loc[:, ~sensor_mask]

    plot_data = X_latent_test.iloc[:, selection]
    ncols = plot_data.shape[1] // 2
    f, ax = plt.subplots(
        nrows=2, ncols=ncols, figsize=(DOUBLE_COLUMN * 0.8, ONEHALF_COLUMN * 0.8)
    )
    f.subplots_adjust(left=0.148, bottom=0.155, right=0.978, top=0.921, hspace=0.248)

    vmin = None
    vmax = None

    for i, col in enumerate(plot_data.columns):

        values = plot_data.loc[:, [col]].to_numpy()
        values = values.reshape(plot_data.n_timeseries, plot_data.n_timesteps)
        values = values[:100, :]

        _ax = ax[i // ncols][np.mod(i, ncols)]

        _ax.imshow(values, vmin=vmin, vmax=vmax, aspect="auto")
        _ax.set_title(i + 1)

        is_first_in_row = np.mod(i, ncols) == 0
        is_first_row = i // ncols == 0

        dmap_idx = int(col.replace("dmap", ""))
        _ax.set_title(
            rf"$\varphi_{{{dmap_idx+1}}}$",
            verticalalignment="center",
            y=1.05,
        )

        if is_first_in_row and is_first_row:
            ax[0][0].set_yticks([0, 25, 50, 75, 100])
            ax[1][0].set_yticks([0, 25, 50, 75, 100])

            f.add_subplot(111, frame_on=False)
            plt.tick_params(labelcolor="none", bottom=False, left=False)
            plt.xlabel("time [hour]")
            plt.ylabel("initial condition")

        if is_first_in_row:
            if is_first_row:
                _ax.set_xticks([])
            else:
                _ax.set_xticks([0, 24])

            _ax.set_yticks([0, 25, 50, 75, 100])
        else:
            if is_first_row:
                _ax.set_xticks([])
            else:
                _ax.set_xticks([0, 24])
            _ax.set_yticks([])
            _ax.set_yticks([])


def plot_paper_dmap_3d(
    X_latent_train,
    X_latent_test,
    X_latent_interp_test,
):
    dmap_selection = np.array([1, 2, 9])  # in paper 1-based indexing, i.e. (2,3,10)

    sensor_mask = X_latent_train.columns.str.startswith("sensor_")
    X_latent_test = X_latent_test.loc[:, ~sensor_mask]

    plot_data_interp = X_latent_interp_test.iloc[:, dmap_selection]
    plot_data_interp = plot_data_interp.drop(plot_data_interp.initial_states(10).index)

    plot_data = X_latent_test.iloc[:, dmap_selection]

    time_idx_interp = pd.DatetimeIndex(plot_data_interp.index.get_level_values("time"))

    from matplotlib.colors import ListedColormap, Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from pylab import cm

    vmax = 1440
    new_cmap = cm.get_cmap("twilight", vmax)

    color_points = lambda tidx: 60 * tidx.hour + tidx.minute

    f = plt.figure(figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN * 0.45))
    f.subplots_adjust(left=0.033, bottom=0.23, top=1, wspace=0.4)
    ax = f.add_subplot(121, projection="3d")
    scatter_point_cloud = ax.scatter(
        plot_data.iloc[:, 0],
        plot_data.iloc[:, 1],
        plot_data.iloc[:, 2],
        c=color_points(plot_data.index.get_level_values("time")),
        s=0.5,
        cmap=new_cmap,
    )

    lab = lambda n: rf"$\varphi_{{{n+1}}}$"
    ax.set_xlabel(lab(dmap_selection[0]))
    ax.set_ylabel(lab(dmap_selection[1]))
    ax.set_zlabel(lab(dmap_selection[2]))
    ax_first = ax

    # Small central plot:
    _plot_data = pd.DataFrame(plot_data.copy())
    _plot_data.index = pd.MultiIndex.from_arrays(
        [
            plot_data.index.get_level_values("ID"),
            plot_data.index.get_level_values("time").strftime("%w:%a-%H"),
        ]
    )

    mean_phi = _plot_data.groupby("time").mean()

    ax = f.add_axes([0.32, 0.15, 0.3, 0.3], projection="3d")
    ax.plot(
        mean_phi.loc[:, "dmap1"].to_numpy().ravel(),
        mean_phi.loc[:, "dmap2"].to_numpy().ravel(),
        mean_phi.loc[:, "dmap9"].to_numpy().ravel(),
        c="black",
        linewidth=1,
    )

    for weekday in ["Wed", "Sun"]:
        if weekday == "Wed":
            color = "orange"

            ax.scatter(
                mean_phi.loc["3:Wed-20", "dmap1"],
                mean_phi.loc["3:Wed-20", "dmap2"],
                mean_phi.loc["3:Wed-20", "dmap9"],
                s=15,
                c="orange",
            )

        elif weekday == "Sun":
            color = "blue"

            ax.scatter(
                mean_phi.loc["0:Sun-08", "dmap1"],
                mean_phi.loc["0:Sun-08", "dmap2"],
                mean_phi.loc["0:Sun-08", "dmap9"],
                s=15,
                c="blue",
            )
        bool_day = mean_phi.index.str.contains(weekday)

        ax.plot(
            mean_phi.loc[bool_day, "dmap1"].to_numpy().ravel(),
            mean_phi.loc[bool_day, "dmap2"].to_numpy().ravel(),
            mean_phi.loc[bool_day, "dmap9"].to_numpy().ravel(),
            c=color,
            linewidth=1,
        )

    ax.view_init(-ax.elev, ax.azim)
    ax.set_xticks([-0.1, 0, 0.1])
    ax.set_yticks([-0.1, 0, 0.1])
    ax.set_zticks([-0.1, 0, 0.1])
    ax.set_xlim(ax_first.get_xlim())
    ax.set_ylim(ax_first.get_xlim())
    ax.set_zlim(ax_first.get_xlim())
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    cax = f.add_axes([0.2, 0.1, 0.6, 0.03])
    cbar = plt.colorbar(
        scatter_point_cloud,
        cax=cax,
        orientation="horizontal",
        ticks=[0, 8 * 60, 16 * 60, 23 * 60 - 1],
    )
    cbar.ax.set_xticklabels(["0 am", "8 am", "4 pm", "12 pm"])

    ax = f.add_subplot(122, projection="3d")

    for _id, _df in plot_data_interp.itertimeseries():
        _plot_data = _df.to_numpy()

        points = _plot_data.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        coll = Line3DCollection(
            segments, norm=Normalize(vmin=0, vmax=vmax), cmap=new_cmap
        )
        coll.set_array(color_points(_df.index))
        ax.add_collection(coll)

    ax.set_xlim(ax_first.get_xlim())
    ax.set_ylim(ax_first.get_ylim())
    ax.set_zlim(ax_first.get_zlim())
    ax.set_xlabel(lab(dmap_selection[0]))
    ax.set_ylabel(lab(dmap_selection[1]))
    ax.set_zlabel(lab(dmap_selection[2]))


def plot_paper_koop_eigval(edmd, select_plot):

    print(f"Largest Koopman eigenvalue {np.max(np.abs(edmd.koopman_eigenvalues))}")

    f, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN))
    f.subplots_adjust(left=0.248, bottom=0.174, top=0.952, right=0.976)

    ax.scatter(
        np.real(edmd.koopman_eigenvalues),
        np.imag(edmd.koopman_eigenvalues),
        s=20,
        marker="o",
        c="white",
        edgecolors="black",
    )

    ax.scatter(
        np.real(edmd.koopman_eigenvalues)[select_plot],
        np.imag(edmd.koopman_eigenvalues)[select_plot],
        s=50,
        edgecolors="black",
        linewidths=0.5,
        c="red",
    )

    circle_values = np.linspace(0, 2 * np.pi, 3000)
    ax.plot(np.cos(circle_values), np.sin(circle_values), "-", color="gray")

    with plt.rc_context(rc={"text.usetex": True}):
        ax.set_xlabel("$\\Re(\\lambda)$")
        ax.set_ylabel("$\\Im(\\lambda)$")

    ax.grid()
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])

    f, ax = plt.subplots(figsize=(SINGLE_COLUMN, SINGLE_COLUMN))
    f.subplots_adjust(left=0.23, bottom=0.171, right=0.976, top=0.955)

    td = lambda d: np.timedelta64(d, "h")
    time_values = np.arange(td(0), td(25), td(1))

    ax = plot_eigenvalues_time(
        time_values,
        eigenvalues=edmd.koopman_eigenvalues,
        delta_time=edmd.dt_,
        ax=ax,
        plot_kwargs=dict(linewidth=0.3, alpha=0.5),
    )

    ax = plot_eigenvalues_time(
        time_values=time_values,
        eigenvalues=edmd.koopman_eigenvalues[select_plot],
        delta_time=edmd.dt_,
        plot_kwargs=dict(color="red", linewidth=0.7),
        ax=ax,
    )

    ax.set_ylim([0, 1.02])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel("time ($t$) [hour]")
    ax.set_xticks([0, 8, 16, 24])
    ax.set_xlim([0, 24.1])
    ax.grid()


def plot_paper_koop_eigfunc(
    edmd,
    selection,
    X_eigfunc_test,
):

    ncols = len(selection)

    f, ax = plt.subplots(
        nrows=2,
        ncols=ncols,
        sharey=True,
        sharex=True,
        figsize=(DOUBLE_COLUMN * 0.8, ONEHALF_COLUMN * 0.8),
    )

    f.subplots_adjust(left=0.164, bottom=0.155, right=0.978, top=0.898, hspace=0.248)

    for i, eigidx in enumerate(selection):
        tsc_data = X_eigfunc_test.iloc[:, [eigidx]]
        eigval = edmd.koopman_eigenvalues[eigidx]

        plot_data = tsc_data.to_numpy()
        plot_data = plot_data.reshape(tsc_data.n_timeseries, tsc_data.n_timesteps)
        plot_data = plot_data[:101, :]

        real_values = np.real(plot_data)
        imag_values = np.imag(plot_data)

        vmin = np.min([np.min(real_values), np.min(imag_values)])
        vmax = np.max([np.max(real_values), np.max(imag_values)])

        ax[0][i].imshow(real_values, vmin=vmin, vmax=vmax, aspect="auto")
        ax[1][i].imshow(imag_values, vmin=vmin, vmax=vmax, aspect="auto")

        if i == 0:
            ax[0][0].set_yticks([0, 25, 50, 75, 100])
            ax[1][0].set_yticks([0, 25, 50, 75, 100])

            ax[0][0].set_ylabel("$\\Re$", rotation=0)
            ax[1][0].set_ylabel("$\\Im$", rotation=0)

            f.add_subplot(111, frame_on=False)
            plt.tick_params(labelcolor="none", bottom=False, left=False)
            plt.xlabel("time [hour]")
            plt.ylabel("initial condition", labelpad=10)

        ax[1][i].set_xticks([0, 24])
        ax[0][i].set_title(f"$\\xi_{{{selection[i]+1}}}$ \n ({eigval:.2f})", fontsize=8)


def plot_dmap_eigenfunctions_imshow(
    edmd,
    X_windows_train,
    X_latent_train,
    X_latent_interp_train,
    X_windows_test,
    X_latent_test,
    X_latent_interp_test,
):

    from sklearn.utils import resample

    sensor_mask = X_latent_train.columns.str.startswith("sensor_")
    X_latent_test = X_latent_test.loc[:, ~sensor_mask]

    # time_idx = pd.DatetimeIndex(X_latent_test.index.get_level_values("time"))
    # time_idx_day = 60 * time_idx.hour + time_idx.minute
    # time_idx_week = 1440 * time_idx.weekday + 60 * time_idx.hour + time_idx.minute

    # local_regress = LocalRegressionSelection(intrinsic_dim=10, n_subsample=500).fit(
    #     X_latent_test.iloc[1:, 1:100].to_numpy()
    # )
    #
    # dmap_selection = local_regress.evec_indices_
    # plot_data = X_latent_test.iloc[:, dmap_selection]
    #
    # plot_pairwise_eigenvector(
    #     plot_data.to_numpy(),
    #     n=0,
    #     scatter_params={"c": time_idx_day, "cmap": "twilight"},
    # )
    # plot_pairwise_eigenvector(
    #     plot_data.to_numpy(), n=0, scatter_params={"c": time_idx_week, "cmap": "hsv"}
    # )

    perform_local_regress = False
    if perform_local_regress:
        local_regress = LocalRegressionSelection(intrinsic_dim=4, n_subsample=500).fit(
            X_latent_test.iloc[:, 1:20].to_numpy()
        )

        dmap_selection = local_regress.evec_indices_
        plot_data = X_latent_test.iloc[:, dmap_selection]

        # remove points too close to the initial condition (-> often outliers)
        # plot_data = plot_data.drop(plot_data.initial_states(10).index)

        time_idx = pd.DatetimeIndex(plot_data.index.get_level_values("time"))
        time_idx_day = 60 * time_idx.hour + time_idx.minute
        time_idx_day = time_idx.hour
        time_idx_week = 1440 * time_idx.weekday + 60 * time_idx.hour + time_idx.minute
        time_idx_dayofweek = time_idx.dayofweek

        plot_data = plot_data.to_numpy()
        (
            plot_data,
            time_idx_day_sub,
            time_idx_week_sub,
            time_idx_dayofweek_sub,
        ) = resample(
            plot_data,
            time_idx_day,
            time_idx_week,
            time_idx_dayofweek,
            n_samples=plot_data.shape[0],  # 10000,  #
        )

        f = plt.figure()
        ax = f.add_subplot(121, projection="3d")

        from pylab import cm

        ax.scatter(
            plot_data[:, 0],
            plot_data[:, 1],
            plot_data[:, 2],
            c=time_idx_day_sub,
            cmap=cm.get_cmap("twilight", 24),
        )
        ax.set_title("color=daytime")
        ax.set_xlabel(dmap_selection[0])
        ax.set_ylabel(dmap_selection[1])
        ax.set_zlabel(dmap_selection[2])

        ax = f.add_subplot(122, projection="3d")
        ax.scatter(
            plot_data[:, 0],
            plot_data[:, 1],
            plot_data[:, 2],
            c=time_idx_dayofweek_sub,
            cmap=cm.get_cmap("tab10", 7),
        )
        ax.set_title("color=weektime")
        ax.set_xlabel(dmap_selection[0])
        ax.set_ylabel(dmap_selection[1])
        ax.set_zlabel(dmap_selection[2])

        f, ax = plt.subplots(ncols=3)
        day = ax[0].scatter(
            plot_data[:, 0],
            plot_data[:, 1],
            c=time_idx_day_sub,
            cmap=cm.get_cmap("tab20b", 24),
        )
        plt.colorbar(day)
        week = ax[1].scatter(
            plot_data[:, 0],
            plot_data[:, 2],
            c=time_idx_dayofweek_sub,
            cmap=cm.get_cmap("tab10", 7),
        )
        plt.colorbar(week)
        week = ax[2].scatter(
            plot_data[:, 0],
            plot_data[:, 3],
            c=time_idx_dayofweek_sub,
            cmap=cm.get_cmap("tab10", 7),
        )

        # plot_pairwise_eigenvector(plot_data.to_numpy(), n=0, scatter_params={"c": time_idx_day })

    # X_latent_interp_test = X_latent_interp_test.iloc[:, dmap_selection]

    # TODO: return DatetimeIndex in time_values() if datetime in TSCDataFrame (> more
    #  convenient)
    # Take away the first 10 minutes because they can be corrupted by noise
    X_latent_interp_test = X_latent_interp_test.drop(
        X_latent_interp_test.initial_states(10).index
    )

    time_idx = pd.DatetimeIndex(X_latent_test.index.get_level_values("time"))
    time_idx_day = 60 * time_idx.hour + time_idx.minute
    time_idx_week = 1440 * time_idx.weekday + 60 * time_idx.hour + time_idx.minute

    plot_data = X_latent_test.iloc[:, 1:]

    temporal_coord = plot_data.iloc[:, [0, 1, 2]].to_numpy()
    temporal_coord, time_idx_day_sub = resample(
        temporal_coord, time_idx_day, n_samples=5000
    )

    f = plt.figure()
    ax = f.add_subplot(projection="3d")
    ax.scatter(
        temporal_coord[:, 0],
        temporal_coord[:, 1],
        temporal_coord[:, 2],
        c=time_idx_day_sub,
        cmap="twilight_shifted",
    )

    # f = plt.figure()
    # ax = f.add_subplot(projection="3d")
    # ax.scatter(X_latent_interp_test.iloc[:, 2], X_latent_interp_test.iloc[:, 3],
    #            X_latent_interp_test.iloc[:, 4], c=time_idx, cmap="twilight_shifted")

    spatial_coord = plot_data.iloc[:, [1, 7, 8]].to_numpy()
    spatial_coord, time_idx_week_sub = resample(
        spatial_coord, time_idx_week, n_samples=5000
    )

    f = plt.figure()
    ax = f.add_subplot(121, projection="3d")
    ax.scatter(
        spatial_coord[:, 0],
        spatial_coord[:, 1],
        spatial_coord[:, 2],
        c=time_idx_week_sub,
        cmap="hsv",
    )

    ax = f.add_subplot(122, projection="3d")
    ax.scatter(
        spatial_coord[:, 0],
        spatial_coord[:, 1],
        spatial_coord[:, 2],
        c=time_idx_day_sub,
        cmap="twilight",
    )

    plot_data = X_latent_test.iloc[:, np.arange(60)]
    f, ax = plt.subplots(nrows=2, ncols=plot_data.shape[1] // 2)

    vmin = None
    vmax = None

    for i, eigidx in enumerate(range(plot_data.shape[1])):
        values = plot_data.iloc[:, [eigidx]].to_numpy()

        values = values.reshape(plot_data.n_timeseries, plot_data.n_timesteps)

        ax[i // 30][np.mod(i, 30)].imshow(values, vmin=vmin, vmax=vmax, aspect="auto")
        ax[i // 30][np.mod(i, 30)].set_title(i + 1)


def plot_koopman_eigenfunctions_imshow(
    edmd,
    X_eigfunc_test,
):

    f, ax = plt.subplots(nrows=2, ncols=20)

    for i in range(20):

        eigval = edmd.koopman_eigenvalues[i]
        values = X_eigfunc_test.iloc[:, [i]].to_numpy()

        values = values.reshape(X_eigfunc_test.n_timeseries, X_eigfunc_test.n_timesteps)

        real_values = np.real(values)
        imag_values = np.imag(values)

        vmin = np.min([np.min(real_values), np.min(imag_values)])
        vmax = np.max([np.max(real_values), np.max(imag_values)])

        ax[0][i].imshow(real_values, vmin=vmin, vmax=vmax, aspect="auto")
        ax[1][i].imshow(imag_values, vmin=vmin, vmax=vmax, aspect="auto")

        ax[0][i].set_title(f"{np.log(eigval):.2f}")


def create_plots(
    edmd,
    X_original: TSCDataFrame,
    X_reconstruct_train,
    X_windows_train,
    X_latent_train,
    X_eigfunc_train,
    X_reconstruct_test,
    X_windows_test,
    X_latent_test,
    X_latent_interp_test,
    X_eigfunc_test,
):

    make_error_table = True
    if make_error_table:
        plot_error_table(
            edmd=edmd,
            X_windows_train=X_windows_train,
            X_reconstruct_train=X_reconstruct_train,
            X_windows_test=X_windows_test,
            X_reconstruct_test=X_reconstruct_test,
        )

    is_plot_time_series = True
    if is_plot_time_series:

        plot_paper_week_timeseries(
            X_windows_test=X_windows_test, X_reconstruct_test=X_reconstruct_test
        )

        # plot_sensor_time_series(
        #     X_original=X_original,
        #     X_reconstruct_train=X_reconstruct_train,
        #     diff_train=diff_train,
        #     initial_states_train=initial_states_train,
        #     X_reconstruct_test=X_reconstruct_test,
        #     diff_test=diff_test,
        #     initial_states_test=initial_states_test,
        # )

    is_plot_imshow_sensors = True

    if is_plot_imshow_sensors:
        plot_paper_sensor_profile(
            X_windows_train=X_windows_train,
            X_reconstruct_train=X_reconstruct_train,
            X_windows_test=X_windows_test,
            X_reconstruct_test=X_reconstruct_test,
        )

        # plot_imshow_sensors(
        #     X_original=X_original,
        #     X_windows_train=X_windows_train,
        #     X_reconstruct_train=X_reconstruct_train,
        #     diff_train=diff_train,
        #     initial_states_train=initial_states_train,
        #     X_windows_test=X_windows_test,
        #     X_reconstruct_test=X_reconstruct_test,
        #     diff_test=diff_test,
        #     initial_states_test=initial_states_test,
        # )

    is_plot_dmap_eigfunc = True
    if is_plot_dmap_eigfunc:
        # plot_dmap_eigenfunctions_imshow(
        #     edmd,
        #     X_windows_train=X_windows_train,
        #     X_latent_train=X_latent_train,
        #     X_latent_interp_train=X_latent_interp_train,
        #     X_windows_test=X_windows_test,
        #     X_latent_test=X_latent_test,
        #     X_latent_interp_test=X_latent_interp_test,
        # )
        plot_timeseries_dmap(
            X_windows_train=X_windows_train,
            X_windows_test=X_windows_test.loc[X_reconstruct_test.index, :],
            X_latent_train=X_latent_train,
            X_latent_test=X_latent_test,
        )

        plot_paper_dmap_selection(
            edmd,
            X_latent_test=X_latent_test,
        )

        plot_paper_dmap_3d(
            X_latent_train=X_latent_train,
            X_latent_test=X_latent_test,
            X_latent_interp_test=X_latent_interp_test,
        )

    is_plot_koop_eigfunc = True

    # _, indices_importance = _importance_eigfunc(edmd, X_windows_train, X_latent_train)
    # indices_importance = indices_importance.ravel()
    selection = np.array([1, 4, 6, 8, 14])

    if is_plot_koop_eigfunc:

        plot_koopman_eigenfunctions_imshow(
            edmd=edmd,
            X_eigfunc_test=X_eigfunc_test,
        )

        plot_paper_koop_eigfunc(
            edmd=edmd,
            selection=selection,
            X_eigfunc_test=X_eigfunc_test,
        )

        # koopman_matrix = np.real(
        #     edmd.dmd_model.eigenvectors_right_
        #     @ np.diag(edmd.dmd_model.eigenvalues_)
        #     @ edmd.dmd_model.eigenvectors_left_
        # )
        #
        # plt.matshow(
        #     koopman_matrix,
        #     vmin=-1,
        #     vmax=1,
        # )

    is_plot_koop_eigval = True
    if is_plot_koop_eigval:
        # Koopman analysis:
        plot_paper_koop_eigval(
            edmd=edmd,
            select_plot=selection,
        )

    plt.show()


def reconstruct_prediction_horizon(
    edmd, X_true, prefix, interp_min_values=None, qois=None
):
    X_reconstruct, X_windows = edmd.reconstruct(
        X=X_true, qois=qois, return_X_windows=True
    )

    starting_times = X_reconstruct.initial_states().index.get_level_values("time").hour
    assert (starting_times == 0).all()

    print(
        f"score {prefix}: {edmd._score_eval(X_windows.loc[X_reconstruct.index, :], X_reconstruct)}"
    )

    X_latent = edmd.transform(X_windows)
    X_eigfunc = edmd.koopman_eigenfunction(X_windows)

    if interp_min_values is not None:
        X_latent_interp = []
        allic = X_windows.initial_states(n_samples=edmd.n_samples_ic_)

        qois_latent = edmd.feature_names_out_.str.startswith("dmap")
        for i, ic in allic.groupby("ID"):
            t_evals = np.repeat(
                ic.time_values()[-1], 1440 // interp_min_values
            ) + np.cumsum(
                np.repeat(
                    np.timedelta64(interp_min_values, "m"), 1440 // interp_min_values
                )
            )
            ic_latent = edmd.transform(ic)

            _X_pred_latent = edmd.dmd_model.predict(ic_latent, t_evals)
            _X_pred_latent = _X_pred_latent.loc[:, qois_latent]
            X_latent_interp.append(_X_pred_latent)

        X_latent_interp = TSCDataFrame.from_frame_list(X_latent_interp)

    else:
        X_latent_interp = None

    return X_reconstruct, X_windows, X_latent, X_latent_interp, X_eigfunc


def run_and_plot_edmd_train_test(edmd, X, n_samples_ic):

    train_time_values, test_time_values = train_test_split(
        X.index.get_level_values("time"),
        train_size=2 / 3.0,
        shuffle=False,
    )

    train_sensor_data = X.loc[pd.IndexSlice[:, train_time_values], :]
    test_sensor_data = X.loc[pd.IndexSlice[:, test_time_values], :]

    print(f"Total n_samples: {train_sensor_data.shape[0] + test_sensor_data.shape[0]}")
    print(f"Total samples train: {train_sensor_data.shape[0]}")
    print(f"Total samples test: {test_sensor_data.shape[0]}")
    print(
        f"Total time series: "
        f"{train_sensor_data.n_timeseries + test_sensor_data.n_timeseries}"
    )
    print(f"Time series train: {train_sensor_data.n_timeseries}")
    print(f"Time series test: {test_sensor_data.n_timeseries}")

    train_sensor_data = filter_data(
        X=train_sensor_data,
        start_time=0,
        n_samples_ic=n_samples_ic,
        min_timesteps=2 * n_samples_ic,
    )

    test_sensor_data = filter_data(
        X=test_sensor_data,
        start_time=0,
        n_samples_ic=n_samples_ic,
        min_timesteps=2 * n_samples_ic,
    )

    plot_paper_data(train_sensor_data, test_sensor_data, n_samples_ic)

    edmd.fit(train_sensor_data, {"dmd__store_system_matrix": True})

    assert (
        edmd.n_samples_ic_ == n_samples_ic
    ), f"true={edmd.n_samples_ic_} set={n_samples_ic}"

    if not use_cache:
        (
            X_reconstruct_train,
            X_windows_train,
            X_latent_train,
            X_latent_interp_train,
            X_eigfunc_train,
        ) = reconstruct_prediction_horizon(
            edmd, train_sensor_data, prefix="train", interp_min_values=None
        )

        pd.DataFrame(X_reconstruct_train).to_csv("X_reconstruct_train.csv")
        pd.DataFrame(X_windows_train).to_csv("X_reconstruct_train.csv")
        pd.DataFrame(X_latent_train).to_csv("X_latent_train.csv")
        pd.DataFrame(X_eigfunc_train).to_csv("X_eigfunc_train.csv")
    else:
        X_reconstruct_train = TSCDataFrame.from_csv(
            "X_reconstruct_train.csv", parse_dates=True
        )
        X_windows_train = TSCDataFrame.from_csv(
            "X_reconstruct_train.csv", parse_dates=True
        )
        X_latent_train = TSCDataFrame.from_csv("X_latent_train.csv", parse_dates=True)
        X_eigfunc_train = TSCDataFrame.from_csv("X_eigfunc_train.csv", parse_dates=True)

    if not use_cache:

        (
            X_reconstruct_test,
            X_windows_test,
            X_latent_test,
            X_latent_interp_test,
            X_eigfunc_test,
        ) = reconstruct_prediction_horizon(
            edmd, test_sensor_data, prefix="test", interp_min_values=30
        )

        pd.DataFrame(X_reconstruct_test).to_csv("X_reconstruct_test.csv")
        pd.DataFrame(X_windows_test).to_csv("X_windows_test.csv")
        pd.DataFrame(X_latent_test).to_csv("X_latent_test.csv")
        pd.DataFrame(X_latent_interp_test).to_csv("X_latent_interp_test.csv")
        pd.DataFrame(X_eigfunc_test).to_csv("X_eigfunc_test.csv")

    else:
        X_reconstruct_test = TSCDataFrame.from_csv(
            "X_reconstruct_test.csv", parse_dates=True
        )
        X_windows_test = TSCDataFrame.from_csv("X_windows_test.csv", parse_dates=True)
        X_latent_test = TSCDataFrame.from_csv("X_latent_test.csv", parse_dates=True)
        X_latent_interp_test = TSCDataFrame.from_csv(
            "X_latent_interp_test.csv", parse_dates=True
        )
        X_eigfunc_test = TSCDataFrame.from_csv("X_eigfunc_test.csv", parse_dates=True)

    print("START PLOTTING")
    create_plots(
        edmd=edmd,
        X_original=X,
        X_reconstruct_train=X_reconstruct_train,
        X_windows_train=X_windows_train,
        X_latent_train=X_latent_train,
        X_eigfunc_train=X_eigfunc_train,
        X_reconstruct_test=X_reconstruct_test,
        X_windows_test=X_windows_test,
        X_latent_test=X_latent_test,
        X_latent_interp_test=X_latent_interp_test,
        X_eigfunc_test=X_eigfunc_test,
    )


if __name__ == "__main__":

    use_cache = False

    if use_cache:
        X = TSCDataFrame.from_csv("X_select_cache.csv", parse_dates=True)
    else:
        X = read_and_select_data()
        X.to_csv("X_select_cache.csv")

    edmd, n_samples_ic = setup_basic_edmd()

    run_and_plot_edmd_train_test(
        edmd,
        X=X,
        n_samples_ic=n_samples_ic,
        use_cache=use_cache,
    )

    plt.show()
