from os.path import dirname, join
from default_config import basic_cfg
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    RandFlipd,
    RepeatChanneld,
    Resized,
    Transposed,
)

cfg = basic_cfg

# data
cfg.data_df = "train_df_4folds.csv"
# 1024 dataset is from:
# https://www.kaggle.com/code/theoviel/dicom-resized-png-jpg
cfg.root_dir = join(dirname(dirname(__file__)), "train_images_png_kaggle")
# train
cfg.train = True
cfg.eval = True
cfg.eval_epochs = 1
cfg.start_eval_epoch = 0
cfg.run_tta_val = False
cfg.amp = True
cfg.val_amp = False
cfg.lr = 3e-4
cfg.lr_div = 1.0
cfg.lr_final_div = 10000.0
cfg.weight_decay = 1e-2
cfg.epochs = 10
# cfg.warmup_epoch = 1
cfg.num_workers = 2
cfg.restart_epoch = 100

# dataset
cfg.img_size = (1024, 1024)
cfg.batch_size = 2
cfg.val_batch_size = 128
cfg.train_cache_rate = 0.0
cfg.val_cache_rate = 0.0
cfg.gpu_cache = False
cfg.val_gpu_cache = False

# model
cfg.backbone = "tf_efficientnetv2_s"
cfg.output_dir = "./output_efnv2s/"
cfg.num_classes = 1
cfg.pos_weight = 1.0
cfg.in_channels = 3
cfg.clf_threshold = 0.1
cfg.drop_rate = 0.2
cfg.drop_path_rate = 0.0

# transforms
cfg.train_transforms = Compose(
    [
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        RepeatChanneld(keys="image", repeats=3),
        Transposed(keys="image", indices=(0, 2, 1)),
        Resized(keys="image", spatial_size=cfg.img_size, mode="bilinear"),
        Lambdad(keys="image", func=lambda x: x / 255.0),
        RandFlipd(keys="image", prob=0.5, spatial_axis=[1]),
    ]
)

cfg.val_transforms = Compose(
    [
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        RepeatChanneld(keys="image", repeats=3),
        Transposed(keys="image", indices=(0, 2, 1)),
        Resized(keys="image", spatial_size=cfg.img_size, mode="bilinear"),
        Lambdad(keys="image", func=lambda x: x / 255.0),
    ]
)
