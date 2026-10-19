"""Arm no_lowcard_all: final minus its all-time cumulative count/stat columns on LOW-cardinality keys.

Cardinality cut: a key is low-cardinality when it has far fewer than ~500 distinct values over the data,
so its all-time running count/mean/std only grows with calendar time (a clock under the time split) and
carries little entity information. Among final's history keys (card1, card1_addr1, uid, uid_email,
P_emaildomain) only P_emaildomain qualifies (59 distinct domains; ProductCD 5, card4 4, card6 4 have no
all-time history columns in final). The others have thousands of distinct values (card1 ~13.5k, the rest
more) and keep their all-time stats.

Dropped (4): the all-time count and the all-time amount statistics of P_emaildomain, including amt_ratio
(= amount / all-time mean). Kept: windowed counts (1d/7d/30d), since_prev, nuniq columns.
Note P_emaildomain_cnt_30d equals cnt_all in the first 30 days only; it is a windowed count and stays.
"""
ARM = {
    "name": "no_lowcard_all",
    "description": "final minus all-time count/amount stats of P_emaildomain (cnt_all, amt_mean, amt_std, amt_ratio)",
    "drop_cols": [
        "P_emaildomain_cnt_all",
        "P_emaildomain_amt_mean",
        "P_emaildomain_amt_std",
        "P_emaildomain_amt_ratio",
    ],
}
