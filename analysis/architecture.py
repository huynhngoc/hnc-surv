from deoxys.model import load_model
from PIL import ImageFont
import visualkeras
import sys
sys.path.append('../')
import customize_obj


font = ImageFont.truetype("C:/Users/Windows User/Downloads/Arial.ttf", 32)

model_path = 'C:/Users/Windows User/Documents/UNI/M30-DV/ulrik/stuk/model.055.h5'
model_test = 'C:/Users/Windows User/Documents/UNI/M30-DV/surv_perf/model/model.002.h5'

model = load_model(model_test).model

font = ImageFont.truetype("arial.ttf", 24)
visualkeras.layered_view(model, legend=True, font=font, to_file='architecture.png').show()
