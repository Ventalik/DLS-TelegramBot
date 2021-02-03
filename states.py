from aiogram.utils.helper import Helper, HelperMode, Item


class BotStates(Helper):
    mode = HelperMode.snake_case

    START_STATE = Item()
    STYLE_TRANSFER_STATE_1 = Item()
    STYLE_TRANSFER_STATE_2 = Item()
    STYLE_GAN_STATE = Item()
    CUBISM_STATE = Item()
    RENAISSANCE_STATE = Item()
    EXPRESSIONISM_STATE = Item()
    SUPER_RESOLUTION_STATE = Item()

