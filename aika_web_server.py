#############################################################################################
# CHANGELOG
# - 20220929 - amoro: trattamento speciale del carattere 'à'
#############################################################################################

# coding=utf8
#import flask
from flask import Flask, request

########################################################################################
# PARTI SPECIFICHE
########################################################################################
from bert import bertRisposta 
from SentimentPackage.src.EmotionClassifier import EmotionClassifier
#from intent_dect_evolution import ClassifyNet 
import SentimentPackage.src.conf as cfg 
import torch
#from keras import backend as K
from sentimentmap import get_mapping_from_sentiment 
import pandas as pd
import time
import json
from datetime import datetime
import traceback
from multiprocessing import Process 

model = None
bertRisp = None
def build():

    '''
    path = "SentimentPackage/models/" + cfg.LOAD_PATH
    global model
    
    model  = EmotionClassifier(4638,num_classes=cfg.N_LABELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path,map_location=device))
    model.eval()
    '''

    global bertRisp
    bertRisp = bertRisposta()
    
build()
########################################################################################

########################################################################################
# INTERFACCIA STANDARD
########################################################################################
app = Flask(import_name=__name__)
@app.route('/aika2_web_app', methods=['GET', 'POST'])
def aika2_web_app():
########################################################################################

    ########################################################################################
    # INPUT STANDARD
    ########################################################################################
    apikey = request.args.get("apikey", "")
    question = request.args.get("question", "")
    extdata = request.args.get("extdata", "")

    print("\naika_web_app: ricevuti i parametri:")
    print("    apikey =", apikey)
    print("    question =", question)
    print("    extdata =", extdata)
    
    # riformattazione stringa ricevuta
    if (question.find("a'") > 0):
        occur_num = question.count("a'")
        rep_question = question.replace("a'", "à", occur_num)
        question = rep_question
        print("    reformatted question =", question)
        
    ########################################################################################
    
    ########################################################################################
    # PARTI SPECIFICHE
    ########################################################################################
    #----------------------------------------------------------------------------
    '''
    start = datetime.now()
    sentiment = model.score_sentence(uuid,"cpu")
    sentiment = get_mapping_from_sentiment(sentiment[0][0])
    end = datetime.now()
    #print("BOT: tempo di calcolo del SENTIMENT:",end-start)
    '''
    sentiment = str()
    
    #-------------------------------------------------------------------------------
    try : 
        startbert=datetime.now()
        risposta, action, action_output, sim_sentiment =bertRisp.gate(question,apikey,False)
        endbert=datetime.now()
        print('\nTEMPO COMPLESSIVO RISPOSTA=', endbert-startbert)
        
        if sim_sentiment:
            sentiment=sim_sentiment

        #registro i LOG
        start_log=datetime.now()
        ip_address=request.remote_addr
        log_text=str(question)+";"+str(risposta.replace('"','').strip())+";"+str(ip_address)
        qa_to_csv=[log_text]
        dati=pd.DataFrame([qa_to_csv])

        #usare per mode 'a' per appendere e 'w' per scrivere senza appendere contenuti
        dati.to_csv('log.csv', mode='a', index=False, header=False)
        end_log=datetime.now()
        print('\nTempo per il salvataggio dei log=', end_log-start_log)
        #-------------------------------------------------------------------------------------
        if action_output!='':
            action=action_output
        print('\n ------------------\n\n')

        # dizionario di ritorno alla pagina web
        resp = risposta
        resp_data = {'action': action, 'action_output': action_output}
        sentim = str()
        dialog = str()


        
    except :
        traceback.print_exc()

        # risposta
        resp = "non lo so mi dispiace"
        resp_data = str()
        sentim = str()
        dialog = str()
    ########################################################################################

    ########################################################################################
    # OUTPUT STANDARD
    ########################################################################################
    # dizionario della risposta alla pagina web (e alla chiamata php)
    web_resp_dict = {'risposta': resp, 'question': question, 'data': resp_data, 'sentiment': sentim, 'dialog': dialog}

    # stringa json di ritorno alla pagina web
    web_resp_str = json.dumps(web_resp_dict, ensure_ascii=False)

    return web_resp_str

   
