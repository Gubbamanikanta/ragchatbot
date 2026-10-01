from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import sqlite3
import pymysql
from transformers import AutoTokenizer, RagSequenceForGeneration
import torch
import faiss
import numpy as np
import pandas as pd
import pickle
import re
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from .medical_kb import (
    DISEASE_KB,
    check_red_flags,
    detect_intent,
    detect_language,
    get_follow_up_questions,
    get_localized_response,
    TRANSLATIONS,
)
from .llm_engine import SymptoTrackLLM

username = "Guest"

tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

MODEL_DIR = os.path.join(settings.BASE_DIR, 'model')
faiss_file = os.path.join(MODEL_DIR, 'faiss.pckl')
y_file = os.path.join(MODEL_DIR, 'Y.npy')

if os.path.exists(faiss_file) and os.path.exists(y_file):
    with open(faiss_file, 'rb') as f:
        faiss_index = pickle.load(f)
    Y = np.load(y_file)
else:
    X = []
    Y = []
    csv_path = os.path.join(settings.BASE_DIR, "Dataset", "Symptom2Disease.csv")
    dataset = pd.read_csv(csv_path, usecols=['label', 'text'])
    dataset = dataset.values
    for i in range(len(dataset)):
        try:
            disease = dataset[i, 0]
            data = dataset[i, 1].strip('\n').strip().lower()
            data = re.sub('[^a-z]+', ' ', data)
            inputs = tokenizer(data, return_tensors="pt")
            input_ids = inputs["input_ids"]
            symptoms_hidden_states = model.question_encoder(input_ids)[0]
            symptoms_hidden_states = symptoms_hidden_states.detach().numpy().ravel()
            X.append(symptoms_hidden_states)
            Y.append(disease)
            print(str(i) + " " + str(len(data)))
        except Exception:
            print(data + "===============================================================")
            pass
    X = np.asarray(X)
    print(X.shape)
    dimension = X.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(X)
    print(faiss_index)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(faiss_file, 'wb') as f:
        pickle.dump(faiss_index, f)
    Y = np.asarray(Y)
    np.save(y_file, Y)


def get_db_connection():
    """
    Resilient DB helper returning (conn, db_type).
    Tries MySQL with blank password, then password 'root', then falls back to SQLite.
    """
    # 1. Try MySQL with blank password
    try:
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', database='symptoms', charset='utf8')
        return con, 'mysql'
    except Exception:
        pass

    # 2. Try MySQL with 'root' password
    try:
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='symptoms', charset='utf8')
        return con, 'mysql'
    except Exception:
        pass

    # 3. Fallback: SQLite
    sqlite_path = os.path.join(settings.BASE_DIR, 'db.sqlite3')
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS register (username TEXT, password TEXT, contact TEXT, email TEXT, address TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS log (username TEXT, question TEXT, advice TEXT, date_time TEXT)")
    con.commit()
    return con, 'sqlite'


def ViewLog(request):
    if request.method == 'GET':
        global username
        output = '<table border=1 align=center width=100%><tr><th><font size="3" color="black">Username</th><th><font size="3" color="black">Symptoms Text</th>'
        output += '<th><font size="3" color="black">Predicted Advice</th><th><font size="3" color="black">Logged & Monitored Time</th></tr>'
        con, db_type = get_db_connection()
        try:
            cur = con.cursor()
            placeholder = '?' if db_type == 'sqlite' else '%s'
            cur.execute(f"SELECT * FROM log WHERE username = {placeholder}", (username,))
            rows = cur.fetchall()
            for row in rows:
                output += '<tr><td><font size="3" color="black">' + str(row[0]) + '</td>'
                output += '<td><font size="3" color="black">' + str(row[1]) + '</td>'
                output += '<td><font size="3" color="black">' + str(row[2]) + '</td>'
                output += '<td><font size="3" color="black">' + str(row[3]) + '</td></tr>'
        finally:
            con.close()
        output += "</table></br></br></br></br>"
        context = {'data': output}
        return render(request, 'UserScreen.html', context)


def logData(username, question, advice):
    now = datetime.now()
    current_datetime = str(now.strftime("%Y-%m-%d %H:%M:%S"))
    con, db_type = get_db_connection()
    try:
        cur = con.cursor()
        placeholder = '?' if db_type == 'sqlite' else '%s'
        sql_query = f"INSERT INTO log VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})"
        cur.execute(sql_query, (str(username), str(question), str(advice), current_datetime))
        con.commit()
    except Exception as e:
        print(f"Error logging data: {e}")
    finally:
        con.close()


def question_encoder_fn(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    return model.question_encoder(input_ids)[0].detach().numpy()


@csrf_exempt
def ChatData(request):
    if request.method == 'GET':
        global tokenizer, faiss_index, Y, username
        question = request.GET.get('mytext', '').strip()
        lang = request.GET.get('lang', '').strip()

        if not question:
            return HttpResponse("Chatbot: Please enter your symptoms.", content_type="text/plain")

        session_id = str(request.session.session_key or username or "default_user")

        response_text, log_entry = SymptoTrackLLM.process_message(
            message=question,
            session_id=session_id,
            lang=lang,
            faiss_index=faiss_index,
            Y=Y,
            question_encoder_fn=question_encoder_fn
        )

        # Only log genuine finalized diagnoses, never greetings or casual conversation!
        if log_entry:
            logData(username, question, log_entry)

        return HttpResponse(response_text, content_type="text/plain; charset=utf-8")


def Chatbot(request):
    if request.method == 'GET':
        return render(request, 'Chatbot.html', {})


def UserScreen(request):
    if request.method == 'GET':
        return render(request, 'UserScreen.html', {})


@csrf_exempt
def UserLoginAction(request):
    global username
    if request.method == 'POST':
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con, db_type = get_db_connection()
        try:
            cur = con.cursor()
            cur.execute("SELECT username, password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if str(row[0]) == str(users) and str(row[1]) == str(password):
                    username = users
                    status = "success"
                    break
        finally:
            con.close()
        if status == 'success':
            context = {'data': 'Welcome ' + username}
            return render(request, "UserScreen.html", context)
        else:
            context = {'data': 'Invalid username'}
            return render(request, 'UserLogin.html', context)


@csrf_exempt
def RegisterAction(request):
    if request.method == 'POST':
        global username
        new_username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)

        output = "none"
        con, db_type = get_db_connection()
        try:
            cur = con.cursor()
            placeholder = '?' if db_type == 'sqlite' else '%s'
            cur.execute(f"SELECT username FROM register WHERE username = {placeholder}", (new_username,))
            rows = cur.fetchall()
            if len(rows) > 0:
                output = str(new_username) + " Username already exists"
            else:
                insert_query = f"INSERT INTO register VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
                cur.execute(insert_query, (new_username, password, contact, email, address))
                con.commit()
                if cur.rowcount >= 1:
                    output = "Signup process completed. Login to perform Symptoms Checker"
        finally:
            con.close()
        context = {'data': output}
        return render(request, 'Register.html', context)


def Register(request):
    if request.method == 'GET':
        return render(request, 'Register.html', {})


def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})


def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})
