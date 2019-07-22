#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from flask import Flask, render_template, redirect, request, url_for, flash
from flask_login import LoginManager, current_user, login_user, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, FileField, TextAreaField
from wtforms.validators import DataRequired

import time
import os
import json
import sys
sys.path.insert(1, '..')
sys.path.insert(2, '../NN/RetinaNet')
import infer_retinanet
from config import Config


IMG_ROOT = 'static/upload'
RESULTS_ROOT = 'static/results'
CORR_RESULTS_ROOT = 'static/corrected'

print("infer_retinanet.BrailleInference()")
t = time.clock()
recognizer = infer_retinanet.BrailleInference()
print(time.clock()-t)

app = Flask(__name__)
app.config.from_object(Config)
login_manager = LoginManager(app)


from flask_uploads import UploadSet, configure_uploads, IMAGES
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = IMG_ROOT
configure_uploads(app, photos)


users_file = 'static/all_users.json'
if os.path.isfile(users_file):
    with open(users_file) as f:
        all_users = json.load(f)
else:
    all_users = dict()

class User:
    def __init__(self, e_mail, name, is_new):
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
        self.e_mail = e_mail
        self.name = name
        if is_new:
            all_users[e_mail] = {'name':name}
            with open(users_file, 'w') as f:
                json.dump(all_users, f, sort_keys=True, indent=4)
    def get_id(self):
        return self.e_mail


@login_manager.user_loader
def load_user(user_id):
    user = all_users.get(user_id, None)
    if user:
        user = User(user_id, user["name"], is_new=False)
    return user

img_path = ""
marked_image_path=""
out_text=""
has_public_confirm = False

@app.route("/", methods=['GET', 'POST'])
@app.route("/index", methods=['GET', 'POST'])
def index():
    global img_path
    global has_public_confirm

    class MainForm(FlaskForm):
        file = FileField("Загрузить файл")
        agree = BooleanField("Я согласен")
        disgree = BooleanField("Возражаю")
        submit = SubmitField("Распознать")
    form = MainForm()
    if form.validate_on_submit():
        if not form.file.data:
            flash('Необходимо загрузить файл')
            return redirect(url_for('index'))
        if form.agree.data and form.disgree.data or not form.agree.data and not form.disgree.data:
            flash('Выберите один из двух вариантов')
            return redirect(url_for('index'))
        os.makedirs(IMG_ROOT, exist_ok=True)
        filename = photos.save(form.file.data)
        img_path = IMG_ROOT + "/" + filename
        has_public_confirm = form.agree.data

        if not form.agree.data:
            return redirect(url_for('confirm'))
        return redirect(url_for('results'))

    return render_template('index.html', current_user = current_user, form=form)


@app.route("/confirm", methods=['GET', 'POST'])
def confirm():
    global img_path
    global has_public_confirm

    class Form(FlaskForm):
        agree = BooleanField("Я согласен")
        disgree = BooleanField("Возражаю", default=True)
        submit = SubmitField('Распознать')

    form = Form()
    if form.validate_on_submit():
        if form.agree.data and form.disgree.data or not form.agree.data and not form.disgree.data:
            flash('Выберите один из двух вариантов')
            return redirect(url_for('index'))
        has_public_confirm = form.agree.data
        return redirect(url_for('results'))
    return render_template('confirm.html', current_user=current_user, form=form, )


@app.route("/results", methods=['GET', 'POST'])
def results():
    global marked_image_path
    global out_text
    extra_info = {'user': current_user.get_id(), 'has_public_confirm': has_public_confirm}
    marked_image_path, out_text = recognizer.run_and_save(img_path, RESULTS_ROOT, lang='RU', extra_info=extra_info,
                                                          draw_refined=recognizer.DRAW_REFINED)  # TODO  accept lang
    return render_template('display.html', filename=marked_image_path, letter=out_text, current_user = current_user)


@app.route("/correct", methods=['GET', 'POST'])
def correct():
    class Form(FlaskForm):
        text = TextAreaField(default='\n'.join(out_text))
        submit = SubmitField('Записать!')
    form = Form()
    if form.validate_on_submit():
        filename_stem = os.path.splitext(os.path.basename(img_path))[0]
        os.makedirs(CORR_RESULTS_ROOT, exist_ok=True)
        path = CORR_RESULTS_ROOT + "/" + filename_stem + '.marked' + '.txt'

        with open(path, 'w') as f:
            f.write(form.text.data)
        flash('СПАСИБО!')
        return redirect(url_for('index'))
    return render_template('correct.html', filename=marked_image_path, form=form, current_user = current_user)


@app.route("/help")
def help():
    return render_template('help.html', current_user = current_user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    class LoginForm(FlaskForm):
        e_mail = StringField('E-mail', validators=[DataRequired()])
        remember_me = BooleanField('Запомнить меня')
        submit = SubmitField('Войти')

    form = LoginForm()
    if form.validate_on_submit():
        user = all_users.get(form.e_mail.data, None)
        if user is None:
            flash('Пользователь не найден. Если Вы - новый пользователь, поставьте галочку выше')
            return redirect(url_for('login'))
        user = User(form.e_mail.data, user['name'], is_new=False)
        #if user is None or not user.check_password(form.password.data):
        #    return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    class RegisterForm(FlaskForm):
        e_mail = StringField('E-mail', validators=[DataRequired()])
        username = StringField('Имя, фамилия', validators=[DataRequired()])
        remember_me = BooleanField('Запомнить меня')
        submit = SubmitField('Зарегистрироваться и войти')

    form = RegisterForm()
    if form.validate_on_submit():
        user = all_users.get(form.e_mail.data, None)
        if user:
            flash('Пользователь с таким E-mail уже существует')
            return redirect(url_for('register'))
        user = User(e_mail=form.e_mail.data, name=form.username.data, is_new=True)
        #if user is None or not user.check_password(form.password.data):
        #    return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))
    return render_template('register.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == "__main__":
    debug = True
    app.jinja_env.cache = {}
    if debug:
        app.run(debug=True)
    else:
        app.run(host='0.0.0.0', threaded=True)
