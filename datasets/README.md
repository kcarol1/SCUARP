# Datasets Summary

下面汇总 `datasets/` 目录中这 4 个数据集的模态信息与数据尺寸。尺寸统一写成 `(H, W, C)`；如果原始数据是二维单通道，则同时注明其原始形状与等价三维形状。

## Data Modalities And Shapes

| 数据集 | 模态 | 文件 | 形状尺寸 |
| --- | --- | --- | --- |
| `2013_DFTC` | CASI 高光谱 | `2013_IEEE_GRSS_DF_Contest_CASI.tif` | `(349, 1905, 144)` |
| `2013_DFTC` | LiDAR | `2013_IEEE_GRSS_DF_Contest_LiDAR.tif` | 原始 `(349, 1905)`，等价于 `(349, 1905, 1)` |
| `Augsburg` | HSI | `data_HS_LR.mat` | `(332, 485, 180)` |
| `Augsburg` | SAR | `data_SAR_HR.mat` | `(332, 485, 4)` |
| `Augsburg` | DSM | `data_DSM.mat` | 原始 `(332, 485)`，等价于 `(332, 485, 1)` |
| `Berlin` | HSI | `data_HS_LR.mat` | `(1723, 476, 244)` |
| `Berlin` | SAR | `data_SAR_HR.mat` | `(1723, 476, 4)` |
| `DFC2007` | Landsat 1994 多光谱 | `Landsat1994.mat` | `(787, 787, 6)` |
| `DFC2007` | Landsat 2000 多光谱 | `Landsat2000.mat` | `(787, 787, 6)` |
| `DFC2007` | ERS 时相影像，共 9 个文件 | `ERS92_aug_13.mat` 等 | 每个文件原始 `(787, 787)`，等价于 `(787, 787, 1)` |

## Label Shapes

| 数据集 | 标签/划分文件 | 形状尺寸 |
| --- | --- | --- |
| `Augsburg` | `GT.mat` (`tainandtest`) | `(332, 485)` |
| `Augsburg` | `TrainImage.mat` | `(332, 485)` |
| `Augsburg` | `TestImage.mat` | `(332, 485)` |
| `Berlin` | `GT.mat` (`tainandtest`) | `(1723, 476)` |
| `Berlin` | `TrainImage.mat` | `(1723, 476)` |
| `Berlin` | `TestImage.mat` | `(1723, 476)` |
| `DFC2007` | `GT.mat` | `(787, 787)` |
| `2013_DFTC` | 训练/验证标注 | 当前目录中为 `roi` / `txt` / `zip` 文件，不是直接存储为单个标签矩阵 |

## Notes

- `2013_DFTC` 的尺寸来自对应 `.hdr` 头文件中的 `lines`、`samples`、`bands`。
- `Augsburg`、`Berlin`、`DFC2007` 的尺寸来自各 `.mat` 文件的变量元信息。
- `Augsburg`、`Berlin`、`DFC2007` 的模态名称主要依据文件名判定，例如 `data_HS_LR`、`data_SAR_HR`、`data_DSM`、`Landsat1994`、`ERS92_aug_13`。
