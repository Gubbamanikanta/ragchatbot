@echo off
echo ========================================================
echo Starting SymptoTrackAI Symptom Monitoring Chatbot...
echo Open your browser and navigate to: http://127.0.0.1:8000/
echo ========================================================
python manage.py runserver 127.0.0.1:8000 --noreload
pause