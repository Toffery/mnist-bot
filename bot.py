import telebot
import model
import conv_model
import config


bot = telebot.TeleBot(config.SECRET_KEY, parse_mode=None)

mnist_model = model.create_model()
conv_mnist_model = conv_model.create_conv_model()


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hi, it's a bot for a digit classification!"
                          "Send me a picture of a digit :)")


@bot.message_handler(content_types=['text'])
def send_answer(message):
    bot.reply_to(message, "I only understand pictures of digits :)")


@bot.message_handler(content_types=['document'])
def handle_docs(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'Received/' + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        # bot.reply_to(message, "Пожалуй, я сохраню это")
        res = conv_model.get_conv_pred(src, conv_mnist_model)
        bot.reply_to(message, f"I guess it's a number: {str(res)}")
    except Exception as e:
        bot.reply_to(message, str(e))


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src = 'Received/' + file_info.file_path
        with open(src, "wb") as new_file:
            new_file.write(downloaded_file)
        # bot.reply_to(message, "Yep")
        res = conv_model.get_conv_pred(src, conv_mnist_model)
        bot.reply_to(message, f"I guess it's a number: {str(res)}")

    except Exception as e:
        bot.reply_to(message, str(e))


bot.infinity_polling()
