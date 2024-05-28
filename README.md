- ./gen_data.py
    - 教師データを生成し, data/train_data.csvとして保存

- ./train.py 
    - 教師データ（data/train_data.csv）を元に学習をおこない, その結果得られた学習済みパラメータをdata/model.pthとして保存

- ./main.py
    - 学習済みパラメータ（data/model.pth）を元にdiffusion modelで円を書くシミュレーションを実行

# diffusion_model

- ./diffusion_model/model.py
    - 機械学習モデルを定義
    - 学習をおこなう関数を定義
    - デノイズをおこなう関数を定義

# simulator

- ./simulator/definition.py
    - シミュレータ内の定数の設定
    - シミュレータのインスタンス生成

- ./simulator/parts.py
    - シミュレータの実装

# train_data_related

- ./train_data_related/gen_data
    - 教師データを生成する関数の実装

- ./train_data_related/input_search.py
    - 山登り法による教師データの生成の実装