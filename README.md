# Football-recognition-bot
Recognition of football players using facial features in a Telegram bot interface.  This project uses a InceptionResnet and MTCNN architectures implemented in Pytorch in the [following github repository](https://github.com/timesler/facenet-pytorch).

# Quick start
1. Run the Download images.ipynb notebook
2. Intially the script downloads 150 images from google images for each of the top 500 best football players in the world. The script selects only 30 of those 500 for classification. In order to alter or add the football players classified, one must change the football_players list in the notebook.
3. Next launch Обучение_модели.ipynb file to fine tune the InceptionResnet model, the model weights are stored inside the model_best.pt file.
4. Replace the telegram key inside the bot.py file and launch it.

# Necessary libraries
1. torchvision==0.9.1
2. torch==1.8.1
3. pandas==1.1.3
4. numpy==1.19.2
5. Pillow==8.0.1
6. matplotlib==3.3.2


