# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
from functools import cmp_to_key
from PIL import Image
import cv2
from numpy import asarray
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, extract_face
from facenet_pytorch import InceptionResnetV1
import torch


def get_faces(img):
    image = Image.open(img)
    if img.endswith('.png') == True:
        image = image.convert('RGB')
    mtcnn = MTCNN(keep_all = True)
    faces = mtcnn(image)
    return faces


def get_predictions(faces, labels_to_faces):
    model = InceptionResnetV1(pretrained='vggface2').eval()
    model.classify = True
    if faces == None:
        return 'Please, send another photo'
    
    if (len(faces) == 3):
        faces = faces.unsqueeze(0)
    img_probs = model(faces)
    _, predicted = torch.max(img_probs, 1)
    predictions = labels_to_faces[predicted]
    msg = "The people in this picture are "
    
    for i in range(len(predictions)):
        msg += predictions[i]
        if i != (len(predictions) - 1):
            msg += ','
    return msg

import logging
import os

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)



def start(update: Update, context: CallbackContext):
    update.message.reply_text('Hi, send an image of a famous person for me to tell you who it is!')





def photo(update: Update, context: CallbackContext):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    update.message.reply_text(
        'Okay now wait a few seconds!!!'
    )
    labels_to_faces = np.load('rcmalli_vggface_labels_v2.npy')
    faces = get_faces('user_photo.jpg')
    
    
    
    update.message.reply_text(get_predictions(faces, labels_to_faces))


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    TOKEN = "1826498400:AAHFd8WwfRHV_gnkl_e5kzY5yLuJnMR-65I"
    updater = Updater(TOKEN, use_context=True)
    PORT = int(os.environ.get('PORT', '88'))

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))
    updater.start_polling()
    updater.idle()

    
if __name__ == '__main__':
    main()
    


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
from functools import cmp_to_key
from PIL import Image
import cv2
from numpy import asarray
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, extract_face
from facenet_pytorch import InceptionResnetV1
import torch

def get_faces(img):
    image = Image.open(img)
    if img.endswith('.png') == True:
        image = image.convert('RGB')
    mtcnn = MTCNN(keep_all = True)
    faces = mtcnn(image)
    return faces


def get_predictions(faces, labels_to_faces):
    model = InceptionResnetV1(pretrained='vggface2').eval()
    model.classify = True
    if faces == None:
        return 'Please, send another photo'
    
    if (len(faces) == 3):
        faces = faces.unsqueeze(0)
    img_probs = model(faces)
    _, predicted = torch.max(img_probs, 1)
    predictions = labels_to_faces[predicted]
    msg = "The people in this picture are "
    
    for i in range(len(predictions)):
        msg += predictions[i]
        if i != (len(predictions) - 1):
            msg += ','
    return msg

import logging
import os

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)



def start(update: Update, context: CallbackContext):
    update.message.reply_text('Hi, send an image of a famous person for me to tell you who it is!')





def photo(update: Update, context: CallbackContext):
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')
    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    update.message.reply_text(
        'Okay now wait a few seconds!!!'
    )
    
    update.message.reply_text(get_predictions('user_photo.jpg'))


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    TOKEN = "1826498400:AAHFd8WwfRHV_gnkl_e5kzY5yLuJnMR-65I" 
    updater = Updater(TOKEN, use_context=True)
    PORT = int(os.environ.get('PORT', '88'))

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, photo))
    updater.idle()

    
if __name__ == '__main__':
    main()
    


