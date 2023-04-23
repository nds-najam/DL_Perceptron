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

def getData(gate):
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
    return DATA


if __name__ == "__main__":
    ETA = 0.1
    EPOCHS = 10
    
    try:
        logging.info(f">>>>> start training for gates <<<<<")
        logging.info(f">>>>> Training for OR gate <<<<<")
        main(data=getData(gate='OR'), modelName="or.model",plotName="or.png", eta=ETA, epochs=EPOCHS)
        logging.info(f">>>>> done training for OR gate <<<<<\n\n")

        logging.info(f">>>>> Training for AND gate <<<<<")
        main(data=getData(gate='AND'), modelName="and.model",plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info(f">>>>> done training for AND gate <<<<<\n\n")

        logging.info(f">>>>> Training for XOR gate <<<<<")
        main(data=getData(gate='XOR'), modelName="xor.model",plotName="xor.png", eta=ETA, epochs=EPOCHS)
        logging.info(f">>>>> done training for XOR gate <<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e