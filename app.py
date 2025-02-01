from flask import Flask, render_template, request, redirect, url_for, session
import random
import re
from the_model import model, tokenizer, pad_sequences, X, lbl_enc, df

app = Flask(__name__)
app.secret_key = 'your_secret_key'  

def generate_answer(user_input):
    """
    Process the user input, generate model predictions, and return a random response.
    """
    if user_input.lower() == 'quit':
        return "Goodbye!"
    
    
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', user_input)
    txt = txt.lower().split()
    txt = " ".join(txt)
    text.append(txt)
    
   
    x_test = tokenizer.texts_to_sequences(text)
    x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])
    
   
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]
    return random.choice(responses)

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'messages' not in session:
        session['messages'] = []

    if request.method == 'POST':
        user_message = request.form.get('message')
        if user_message:
            messages = session.get('messages', [])
            messages.append({'role': 'user', 'content': user_message})
            bot_response = generate_answer(user_message)
            messages.append({'role': 'bot', 'content': bot_response})
            session['messages'] = messages
        return redirect(url_for('index'))
    
    return render_template('index.html', messages=session['messages'])

@app.route('/clear')
def clear():
    session.pop('messages', None)
    return redirect(url_for('index'))

@app.route('/depression-test', methods=['GET', 'POST'])
def depression_test():
    questions = [
        "How often have you felt little interest or pleasure in doing things?",
        "How often have you felt down, depressed, or hopeless?",
        "How often have you had trouble falling asleep, staying asleep, or sleeping too much?",
        "How often have you felt tired or had little energy?",
        "How often have you had a poor appetite or overeating?",
        "How often have you felt bad about yourself or that you are a failure or have let yourself or your family down?",
        "How often have you had trouble concentrating on things, such as reading the newspaper or watching television?",
        "How often have you moved or spoken so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual?",
        "How often have you had thoughts that you would be better off dead or of hurting yourself in some way?"
    ]
    options = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
    option_values = [0, 1, 2, 3]

    if request.method == 'POST':
        responses = []
        for i in range(len(questions)):
            response = request.form.get(f"question_{i}")
            if response not in options:
                response = "Not at all"
            responses.append(option_values[options.index(response)])
        score = sum(responses)

        if score <= 4:
            severity = "Minimal depression"
            advice = [
                "You're doing great! Keep maintaining a positive outlook.",
                "Continue engaging in activities that make you happy.",
                "Keep in touch with friends and loved ones."
            ]
        elif 5 <= score <= 9:
            severity = "Mild depression"
            advice = [
                "Try to stay active and exercise regularly.",
                "Keep a journal to track your thoughts and feelings.",
                "Reach out to friends and family for support."
            ]
        elif 10 <= score <= 14:
            severity = "Moderate depression"
            advice = [
                "Consider talking to a therapist or counselor.",
                "Make time for activities you enjoy.",
                "Practice relaxation techniques such as deep breathing or meditation."
            ]
        elif 15 <= score <= 19:
            severity = "Moderately severe depression"
            advice = [
                "It's important to seek help from a mental health professional.",
                "Try to establish a routine to help manage your day-to-day activities.",
                "Stay connected with supportive friends and family."
            ]
        else:
            severity = "Severe depression"
            advice = [
                "Please seek immediate help from a mental health professional.",
                "Consider reaching out to a crisis hotline or support group.",
                "Remember that you don't have to go through this alone; support is available."
            ]
        return render_template("depression_result.html", score=score, severity=severity, advice=advice)

    return render_template("depression_test.html", questions=questions, options=options)

if __name__ == '__main__':
    app.run(debug=True)
