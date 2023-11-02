# imports 

from flask import Flask, render_template, url_for, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import openai
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#user authentification 

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)



login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')




@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('choose'))
    return render_template('login.html', form=form)



@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


##############################################################################################################################
#  APIs keys

# GPTs key : 
openai.api_key = ''

################################################################################################################################
# the home and choose service pages

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/choose')
@login_required
def choose(): 
    return render_template('choose.html')

#################################################################################################################################
@app.route('/IEEExtreme')
def IEEExtreme() : 
    return render_template('IEEExtreme.html')

#################################################################################################################################

# the generation service : 

criteria1 = f"""the response should be exclusively in French, you should produce no more that 100 words writing, and the writing style sould beclear and comprehensible by a student, and it shall be according to the following topic and/or questions:
while respecting these criterias : 
    - the form is : introduction, body, conclusion (each in a paragraph)
    -conformité de la production à la consigne d'écriture
    -cohérence de l'argumentation
    -Vocabulaire (usage de termes précis et variés)
    -Syntaxe (construction de phrases correctes)
    Conjugaison (emploi des temps)
"""
criteria2 ="no more than 100 words, and at the end of each paragraph make a newline"

def getresponse(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": criteria1 + prompt + criteria2}]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    
    return response.choices[0].message["content"]    

@app.route('/generation', methods=['GET', 'POST'])
@login_required
def generation():
    response_text = ""
    if request.method == 'POST':
        question = request.form['question']
        response_text = getresponse(question)
    return render_template('generation.html', response=response_text)


####################################################################################################################################
# the correction service 

###############################################################################
#trial using pytesseract : 
import pytesseract
import os

# Set the LD_LIBRARY_PATH to include the directory where libarchive.so.13 is located
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib'


from PIL import Image

def ocr_pytesseract(img) : 

    '''
    #Rescale gambar
    width = 350
    height = 250
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Konfigurasi tesseract
    config = "-l eng+jpn+kor+rus+chi_sim+vie+thai"
    '''
    config = "-l eng+jpn+kor+rus+chi_sim+vie+thai"

    #text = pytesseract.image_to_string(thresh, config=config)
    text = pytesseract.image_to_string(img, config=config)
    
    #pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"

    text = pytesseract.image_to_string(img)
    return(text)

    return text 
#

###############################################################################
    

def getcorrection(prompt, model="gpt-3.5-turbo"): 
    messages = [{"role" : "user", "content":  prompt  }]
    response = openai.ChatCompletion.create(
        model = model, 
        messages = messages, 
        temperature = 0, 
    )
    return response.choices[0].message["content"]
 
from io import BytesIO
@app.route('/correction', methods=['GET', 'POST'])
@login_required
def correction():
    text = ''
    prompt = f" Corriger les erreurs grammtical, de syntaxe.. dans ce texte :{text}"


    if request.method == 'POST':
        # Process uploaded file, if it exists
        #file = request.files.get('file')
        #if file and file.filename:
         #   image_content = file.read()
         #   img = Image.open(BytesIO(image_content))

            #text += ocr_google_vision(image_content)
          #  text += ocr_pytesseract(img)

        # Add manually input text, if it exists
        manual_text = request.form.get('text', '')
        text += " " + manual_text  # Add a space between to separate the two inputs
        corrected_text = getcorrection(prompt)
    else:
        corrected_text = ''

    return render_template('correction.html', original=text, corrected=corrected_text)

##################################################################################################################################

# les oeuvres service 


# getting the segments
with open('boite.pkl', 'rb') as f:
    b = pickle.load(f)

with open('condamne.pkl', 'rb') as f:
    c = pickle.load(f)

with open('antigone.pkl', 'rb') as f:
    a = pickle.load(f)

def getanswer(prompt ,model="gpt-3.5-turbo"): 
    messages = [{"role" : "user", "content":  prompt }]
    response = openai.ChatCompletion.create(
        model = model, 
        messages = messages, 
        temperature = 0, 
    )
    return response.choices[0].message["content"]


# getting the best semgent out of all segments 



tfidf_vectorizer_1 = TfidfVectorizer(stop_words='english')
tfidf_matrix_1 = tfidf_vectorizer_1.fit_transform(b)

tfidf_vectorizer_2 = TfidfVectorizer(stop_words='english')
tfidf_matrix_2 = tfidf_vectorizer_2.fit_transform(c)

tfidf_vectorizer_3 = TfidfVectorizer(stop_words='english')
tfidf_matrix_3 = tfidf_vectorizer_3.fit_transform(a)

def get_best_segment_boite(question):
    # Transform the query into the same feature space as the segments
    query_vector = tfidf_vectorizer_1.transform([question])
    # Compute cosine similarity between the question and all segments
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix_1).flatten()
    # Get the index of the most similar segment
    best_segment_index = cosine_similarities.argmax()
    return b[best_segment_index]

def get_best_segment_condamne(question):
    query_vector = tfidf_vectorizer_2.transform([question])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix_2).flatten()
    best_segment_index = cosine_similarities.argmax()
    return c[best_segment_index]

def get_best_segment_antigone(question):
    query_vector = tfidf_vectorizer_3.transform([question])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix_3).flatten()
    best_segment_index = cosine_similarities.argmax()
    return a[best_segment_index]


def ask_boite(question):
    best_segment = get_best_segment_boite(question)
    
    prompt = f"""take a deep breath and use the segment between three dashes - to answer the question, provide you reasoning step by step in French: 
    ---
    {best_segment}
    ---
    {question}
    """
    response = getanswer(prompt)

    return response

def ask_condamne(question):
    best_segment = get_best_segment_condamne(question)
    
    prompt = f"""take a deep breath and use the segment between three dashes - to answer the question, provide you reasoning step by step in French: 
    ---
    {best_segment}
    ---
    {question}
    """
    response = getanswer(prompt)

    return response

def ask_antigone(question):
    best_segment = get_best_segment_antigone(question)
    
    prompt = f"""take a deep breath and use the segment between three dashes - to answer the question in French: 
    ---
    {best_segment}
    ---
    {question}
    """
    response = getanswer(prompt)

    return response

@app.route('/lesoeuvres')
@login_required
def lesoeuvres(): 
    return render_template('lesoeuvres.html')

@app.route('/lesoeuvres/condamne', methods=['GET', 'POST'])
@login_required
def condamne():
    if request.method == 'POST':
        user_message = request.form.get('user_message')
        chat_response = ask_condamne(user_message)
        return render_template('condamne.html', chat_response=chat_response)
    
    return render_template('condamne.html')

@app.route('/lesoeuvres/boite', methods=['GET', 'POST'])
@login_required
def boite():
    if request.method == 'POST':
        user_message = request.form.get('user_message')
        chat_response = ask_boite(user_message)
        return render_template('boite.html', chat_response=chat_response)
    
    return render_template('boite.html')

@app.route('/lesoeuvres/antigone', methods=['GET', 'POST'])
@login_required
def antigone():
    if request.method == 'POST':
        user_message = request.form.get('user_message')
        chat_response = ask_antigone(user_message)
        return render_template('antigone.html', chat_response=chat_response)
    
    return render_template('antigone.html')



if __name__ == '__main__':
    app.run(debug=True)
