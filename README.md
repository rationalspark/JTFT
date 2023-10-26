# JTFT

This is an anonymous implementation of JTFT: "A Joint Time-frequency Domain Transformer for Multivariate Time Series Forecasting."

## Usage

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all data files in the directory.

3. Training. All the scripts are in the directory ```./scripts/```. The scripts can be run using commands such as

```
sh ./scripts/weather.sh
```

The results will be displayed in the log files once the training is completed. The path of the log files will be printed at the beginning of the training.


## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai


## License

Some of the codes are obtained from https://github.com/yuqinie98/PatchTST. These files are licensed under the Apache License Version 2.0.

The new files of JTFT are licensed under the GNU General Public License (GPL) version 2.0. Comments to show the license appears at the beginning of these file.
