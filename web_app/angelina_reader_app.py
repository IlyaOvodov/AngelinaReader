#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
web application Angelina Braille reader
"""
from flask import Flask, render_template, redirect, request, url_for, flash
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, FileField, TextAreaField, HiddenField, SelectField
from wtforms.validators import DataRequired
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_mobility import Mobility
from flask_mobility.decorators import mobile_template

import time
import os
import json
import sys
import argparse
from pathlib import Path
import local_config
import model.infer_retinanet as infer_retinanet
from .config import Config

model_root = 'weights/retina_chars_eced60'
model_weights = '.clr.008'
orientation_attempts = {0,1,4,5}

print("infer_retinanet.BrailleInference()")
t = time.clock()
recognizer = infer_retinanet.BrailleInference(
    params_fn=os.path.join(local_config.data_path, model_root + '.param.txt'),
    model_weights_fn=os.path.join(local_config.data_path, model_root + model_weights),
    create_script=None)
print(time.clock()-t)

app = Flask(__name__)
Mobility(app)
app.config.from_object(Config)
login_manager = LoginManager(app)

IMG_ROOT = Path(app.root_path) / app.config['DATA_ROOT'] / 'raw'
RESULTS_ROOT = Path(app.root_path) / app.config['DATA_ROOT'] / 'results'
os.makedirs(Path(app.root_path) / app.config['DATA_ROOT'], exist_ok=True)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = IMG_ROOT
configure_uploads(app, photos)

users_file = Path(app.root_path) / app.config['DATA_ROOT'] / 'all_users.json'
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
            assert e_mail not in all_users.keys()
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


@app.route("/", methods=['GET', 'POST'])
@app.route("/index", methods=['GET', 'POST'])
@mobile_template('{m/}index.html')
def index(template, is_mobile=False):
    class MainForm(FlaskForm):
        camera_file = FileField()
        file = FileField()
        agree = BooleanField("Я согласен")
        disgree = BooleanField("Возражаю")
        lang = SelectField(choices=[('RU', 'RU'), ('EN', 'EN')])
    form = MainForm()
    if form.validate_on_submit():
        file_data = form.camera_file.data or form.file.data
        if not file_data:
            flash('Необходимо загрузить файл')
            return render_template(template, form=form)
        if form.agree.data and form.disgree.data or not form.agree.data and not form.disgree.data:
            flash('Выберите один из двух вариантов (согласен/возражаю)')
            return render_template(template, form=form)
        os.makedirs(IMG_ROOT, exist_ok=True)
        filename = photos.save(file_data)
        img_path = IMG_ROOT / filename
        has_public_confirm = form.agree.data
        lang = form.lang.data

        if not form.agree.data:
            return redirect(url_for('confirm', img_path=img_path, lang=lang))
        return redirect(url_for('results', img_path=img_path, has_public_confirm=has_public_confirm, lang=lang))

    return render_template(template, form=form)


@app.route("/confirm", methods=['GET', 'POST'])
@login_required
@mobile_template('{m/}confirm.html')
def confirm(template):
    class Form(FlaskForm):
        img_path = HiddenField()
        agree = BooleanField("Я согласен на публикацию.")
        disgree = BooleanField("Возражаю. Это приватный текст.", default=True)
        submit = SubmitField('Распознать')
    form = Form()
    if form.validate_on_submit():
        if form.agree.data and form.disgree.data or not form.agree.data and not form.disgree.data:
            flash('Выберите один из двух вариантов (согласен/возражаю)')
            return render_template(template, form=form)
        has_public_confirm = form.agree.data
        return redirect(url_for('results', img_path=request.form['img_path'], has_public_confirm=has_public_confirm,
                                lang=request.values['lang']))
    form = Form(img_path = request.values['img_path'])
    return render_template(template, form=form)


@app.route("/results", methods=['GET', 'POST'])
@login_required
@mobile_template('{m/}results.html')
def results(template):
    class ResultsForm(FlaskForm):
        marked_image_path = HiddenField()
        text = TextAreaField()
        submit = SubmitField('Записать!')
    form = ResultsForm()
    extra_info = {'user': current_user.get_id(), 'has_public_confirm': request.values['has_public_confirm'],
                  'lang': request.values['lang']}
    marked_image_path, out_text = recognizer.run_and_save(request.values['img_path'], RESULTS_ROOT,
                                                          lang=request.values['lang'], extra_info=extra_info,
                                                          draw_refined=recognizer.DRAW_NONE,
                                                          orientation_attempts=orientation_attempts)
    # convert OS path to flask html path
    root_dir = str(Path(app.root_path))
    marked_image_path = str(Path(marked_image_path))
    assert(marked_image_path[:len(root_dir)]) == root_dir
    marked_image_path = marked_image_path[len(root_dir):].replace("\\", "/")
    out_text = '\n'.join(out_text)
    form = ResultsForm(marked_image_path=marked_image_path,
                       text=out_text)
    return render_template(template, form=form)


@app.route("/help")
def help():
    return render_template('help.html')


@app.route("/results_demo")
@mobile_template('{m/}results_demo.html')
def results_demo(template):
    time.sleep(1)
    return render_template(template)


@app.route('/login', methods=['GET', 'POST'])
@mobile_template('{m/}login.html')
def login(template):
    class LoginForm(FlaskForm):
        e_mail = StringField('E-mail', validators=[DataRequired()])
        remember_me = BooleanField('Запомнить меня')
        submit = SubmitField('Войти')

    form = LoginForm()
    if form.validate_on_submit():
        user = all_users.get(form.e_mail.data, None)
        if user is None:
            flash('Пользователь не найден. Если вы - новый пользователь, зарегистрируйтесь')
            return redirect(url_for('register'))
        user = User(form.e_mail.data, user['name'], is_new=False)
        #if user is None or not user.check_password(form.password.data):
        #    return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))
    return render_template(template, title='Sign In', form=form)


@app.route('/register', methods=['GET', 'POST'])
@mobile_template('{m/}register.html')
def register(template):
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
    return render_template(template, title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


def run():
    parser = argparse.ArgumentParser(description='Angelina Braille reader web app.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='enable debug mode (default: off)')
    args = parser.parse_args()
    debug = args.debug
    if debug:
        print('running in DEBUG mode!')
    else:
        print('running with no debug mode')
    app.jinja_env.cache = {}
    if debug:
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.run(debug=True, port=5001)
    else:
        app.run(host='0.0.0.0', threaded=True)


if __name__ == "__main__":
    run()