from aiogram import types
from main import dp, bot
from PIL import Image
from states import BotStates
from models.inference import get_model
from messages import MESSAGES
import io


@dp.message_handler(content_types=['photo', 'document'], state='*')
async def process_photo(msg: types.Message):
    state = dp.current_state(user=msg.from_user.id)

    model = await get_model(await state.get_state())
    if model is not None:
        image = await load_img_from_message(msg)
        if max(image.size) >= 1280 and state.get_state() == BotStates.SUPER_RESOLUTION_STATE:
            await bot.send_message(msg.from_user.id, MESSAGES['big picture'])
        else:
            styled_image = await model.predict(image)
            photo = await load_img_in_buffer(styled_image)

            await bot.send_document(msg.from_user.id, document=photo)
    else:
        await bot.send_message(msg.from_user.id, MESSAGES['unknown photo'])

    await state.set_state(BotStates.START_STATE)


async def load_img_from_message(msg: types.Message):
    buffer = io.BytesIO()
    if msg.content_type == 'photo':
        await msg.photo[-1].download(buffer)
    else:
        await msg.document.download(buffer)
    image = Image.open(buffer)

    return image


async def load_img_in_buffer(image):
    buffer = io.BytesIO()
    buffer.name = 'output.jpeg'

    pil_image = Image.fromarray(image)
    pil_image.save(buffer, 'jpeg')
    buffered_img = types.InputFile(buffer)
    buffer.seek(0)

    return buffered_img