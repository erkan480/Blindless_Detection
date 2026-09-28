from PIL import Image

file_name = "ai_background.png"

image = Image.open(file_name)

new_size = (1600,900)

resized_image = image.resize(new_size)

resized_image.save(file_name)
