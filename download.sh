mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list

# download data
kaggle datasets download theoviel/rsna-breast-cancer-1024-pngs

mkdir train_images_png_kaggle
unzip -j rsna-breast-cancer-1024-pngs.zip "output/*" -d "train_images_png_kaggle"
rm rsna-breast-cancer-1024-pngs.zip