from data import load_data
from color_model import ColorizationModel

def training():
    DIR = './data/128x128/'
    model = ColorizationModel().define_model()
    X, Y = load_data(DIR)
    model.summary()
    model.fit(X, Y, batch_size = 50, epochs = 200)
    model.save('./trained_model/color_model_image128x128.h5')

if __name__ == "__main__":
    training()
