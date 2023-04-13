from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd
import logging
import os

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir,"running_logs.log"),
    level=logging.INFO,
    format='[%(asctime)s:%(levelname)s:%(module)s]: %(message)s',
    filemode='a'
     )

def main(data,modelName,plotName,eta,epochs):
    df = pd.DataFrame(data)
    logging.info(f"This is the raw dataset: \n{df}")
    X, y = prepare_data(df)
    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)

    _ = model.total_loss()

    model.save(filename = modelName, model_dir='model')
    save_plot(df, model, filename = plotName)

if __name__ == "__main__":
    ETA = 0.1
    EPOCHS = 10
    gate = 'OR'
    if gate == 'OR':
        DATA = {
            "x1": [0,0,1,1],
            "x2": [0,1,0,1],
            "y" : [0,1,1,1]
        }
    elif gate == 'AND':
        DATA = {
            "x1": [0,0,1,1],
            "x2": [0,1,0,1],
            "y" : [0,0,0,1]
        } 
    elif gate == 'XOR':
        DATA = {
            "x1": [0,0,1,1],
            "x2": [0,1,0,1],
            "y" : [0,1,1,0]
        }
    else:
        logging.info("Bad GATE chosen")
    try:
        logging.info(f">>>>> start training for {gate} gate <<<<<")
        main(data=DATA, modelName=gate+".model",plotName=gate+".png", eta=ETA, epochs=EPOCHS)
        logging.info(f">>>>> done training for {gate} gate <<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e