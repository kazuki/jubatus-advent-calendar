import bz2
import json
import time
import os

from jubatus import Classifier
from jubatus.classifier.types import LabeledDatum
from jubatus.common import Datum
from embedded_jubatus import Classifier as EmbeddedClassifier

HOST='127.0.0.1'
PORT=9199
CONFIG='/usr/share/jubatus/example/config/classifier/default.json'


def prepare_data():
    def _load_json(name):
        return json.load(
            open(name, 'r', encoding='utf8') if os.path.exists(name) else
            bz2.open(name + '.bz2', 'rt', encoding='utf8'))
    txt_data, num_data = [], []
    for row in _load_json('../20news.json'):
        txt_data.append(LabeledDatum(row[0], Datum({'body': row[1]})))
    for row in _load_json('../dorothea_train.json'):
        num_data.append(LabeledDatum(row[0], Datum(
            {str(i): int(row[i]) for i in range(1, len(row)) if row[i]})))
    return (txt_data, num_data)


def main():
    txt_data, num_data = prepare_data()

    c = [
        Classifier(HOST, PORT, '', 6000),
        EmbeddedClassifier(CONFIG)
    ]
    n = ['server', 'embedded']

    for (name, data) in zip(('20news', 'dorothea'), (txt_data, num_data)):
        print('# {}'.format(name))
        for bsz in [1, 10, 100, len(data)]:
            print('= train ({} rows/call) ='.format(bsz))
            for i in range(len(c)):
                start_time = time.time()
                for j in range(0, len(data), bsz):
                    c[i].train(data[j:j+bsz])
                end_time = time.time()
                print('  {}={}[ms]'.format(n[i], end_time - start_time))
                c[i].clear()
            print()

main()
