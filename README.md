# rsna
Currently, this is copy from https://www.kaggle.com/code/yiheng/monai-pipeline-training/notebook
solution.

## Downloading data
1. Create kaggle token
   1. `Create new API token` and download `kaggle.json` into root folder.
   2. Go to https://www.kaggle.com/docs/api "Authentication" to see the details.
2. Run shell command `chmod -x downlaod.sh && ./downlaod.sh`
3. Install `requirments.txt`
4. To change the batch_size and num_workers check `configs/cfg_clf_baseline.py:40` and `configs/cfg_clf_baseline.py:35`
5. Run script from root ``