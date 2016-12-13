import csv
import math
import sys

from jubatus.common import Datum
from jubatus.regression.types import ScoredDatum
from embedded_jubatus import Regression


regressor = Regression({
    'converter' : {
        'string_filter_types' : {}, 'string_filter_rules' : [],
        'num_filter_types' : {}, 'num_filter_rules' : [],
        'string_types': {}, 'num_types' : {},
        'string_rules': [
            { 'key' : '*', 'type' : 'space', 'sample_weight' : 'bin', 'global_weight' : 'bin' }
        ],
        'num_rules' : [
            { 'key' : '*', 'type' : 'num' }
        ]
    },
    'parameter' : {
        'sensitivity' : 2,
        'regularization_weight' : 0.25
    },
    'method': 'AROW'
})

# 時刻データ(HH:mm)を時を単位とした小数に変換
parse_time = lambda s: sum([float(t) / (60**i) for i, t in enumerate(s.split(':'))])

# CSVデータを読み込み，特徴量を抽出
data = []
with open('data.csv') as f:
    reader = csv.reader(f)
    prev_target, prev_left = [None] * 2
    for r in reader:
        month = float(r[0])        # 月
        day = r[1]                 # 曜日
        target = parse_time(r[2])  # 出社時間(予測対象)
        left = parse_time(r[3])    # 退社時間
        if prev_target is not None:
            d = Datum({
                'prev_target': prev_target,  # 前日の出社時間
                'prev_left': prev_left,      # 前日の退社時間
                'day': day,                  # 曜日
                'month': month,              # 月
            })
            data.append(ScoredDatum(target, d))
        prev_target = target
        prev_left = left

# JubatusのRegressionを使って学習
regressor.train(data)
with open('model.jubatus', 'wb') as f:
    f.write(regressor.save_bytes())

# 教師データを使ってどの程度の予測誤差があるのか計測
# (教師データを使っているので参考程度に)
y = [x.score for x in data]
pred = regressor.estimate([x.data for x in data])
mae = sum([abs(pred - answer) for pred, answer in zip(pred, y)]) / len(y)
mse = sum([(pred - answer)**2 for pred, answer in zip(pred, y)]) / len(y)
print('MAE={}, RMSE={}'.format(mae, math.sqrt(mse)))

print(regressor.estimate([
    Datum({
        'prev_target': parse_time(sys.argv[3]),
        'prev_left': parse_time(sys.argv[4]),
        'day': sys.argv[2],
        'month': float(sys.argv[1]),
    })
])[0])
