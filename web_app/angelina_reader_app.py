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
from flask_uploads import UploadSet, configure_uploads, IMAGES, ARCHIVES
from flask_mobility import Mobility
from flask_mobility.decorators import mobile_template

from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import smtplib

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

photos = UploadSet('photos', extensions=IMAGES + ('pdf',) + ARCHIVES)

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
        lang = SelectField("Язык текста", choices=[('RU', 'RU'), ('EN', 'EN')])
        find_orientation = BooleanField("Авто ориентация")
        process_2_sides = BooleanField("Обе стороны")
    form = MainForm(agree=request.values.get('has_public_confirm', '') == 'True',
                    disgree=request.values.get('has_public_confirm', '') == 'False',
                    lang=request.values.get('lang', 'RU'),
                    find_orientation=request.values.get('find_orientation', 'False') == 'True',
                    process_2_sides=request.values.get('process_2_sides', 'False') == 'True',
                    )
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

        if not form.agree.data:
            return redirect(url_for('confirm',
                                    img_path=img_path,
                                    lang=form.lang.data,
                                    find_orientation=form.find_orientation.data,
                                    process_2_sides=form.process_2_sides.data))
        return redirect(url_for('results',
                                img_path=img_path,
                                has_public_confirm=form.agree.data,
                                lang=form.lang.data,
                                find_orientation=form.find_orientation.data,
                                process_2_sides=form.process_2_sides.data))

    return render_template(template, form=form)


@app.route("/confirm", methods=['GET', 'POST'])
@login_required
@mobile_template('{m/}confirm.html')
def confirm(template):
    class Form(FlaskForm):
        agree = BooleanField("Я согласен на публикацию.")
        disgree = BooleanField("Возражаю. Это приватный текст.", default=True)
        submit = SubmitField('Распознать')
    form = Form()
    if form.validate_on_submit():
        if form.agree.data and form.disgree.data or not form.agree.data and not form.disgree.data:
            flash('Выберите один из двух вариантов (согласен/возражаю)')
            return render_template(template, form=form)
        has_public_confirm = form.agree.data
        return redirect(url_for('results',
                                img_path=request.values['img_path'],
                                has_public_confirm=has_public_confirm,
                                lang=request.values['lang'],
                                find_orientation=request.values['find_orientation'],
                                process_2_sides=request.values['process_2_sides']))
    return render_template(template, form=form)


@app.route("/results", methods=['GET', 'POST'])
@login_required
@mobile_template('{m/}results.html')
def results(template):
    class ResultsForm(FlaskForm):
        results_list = HiddenField()
        submit = SubmitField('отправить на e-mail')
    form = ResultsForm()
    if form.validate_on_submit():
        return redirect(url_for('email',
                                results_list=request.form['results_list'],
                                has_public_confirm=request.values['has_public_confirm'],
                                lang=request.values['lang'],
                                find_orientation=request.values['find_orientation'],
                                process_2_sides=request.values['process_2_sides']))

    extra_info = {'user': current_user.get_id(), 'has_public_confirm': request.values['has_public_confirm']=='True',
                  'lang': request.values['lang']}
    ext = Path(request.values['img_path']).suffix[1:]  # exclude leading dot
    if ext in IMAGES or ext == 'pdf':
        results_list = recognizer.run_and_save(request.values['img_path'], RESULTS_ROOT,
                                               lang=request.values['lang'], extra_info=extra_info,
                                               draw_refined=recognizer.DRAW_NONE,
                                               remove_labeled_from_filename=False,
                                               find_orientation=request.values['find_orientation']=='True',
                                               process_2_sides=request.values['process_2_sides']=='True')
    # elif ext in ARCHIVES:  TODO
    #     results_list = recognizer.run_and_save_archive(request.values['img_path'], RESULTS_ROOT,
    #                                                           lang=request.values['lang'], extra_info=extra_info,
    #                                                           draw_refined=recognizer.DRAW_NONE,
    #                                                           find_orientation=request.values['find_orientation']=='True',
    #                                          process_2_sides=request.values['process_2_sides']=='True')
    else:
        assert False, "incorrect file type: " + str(request.values['img_path'])
    # convert OS path to flask html path
    root_dir = str(Path(app.root_path))
    image_paths_and_texts = list()
    file_names = list()
    for marked_image_path, recognized_text_path, out_text in results_list:
        flask_image_path = str(Path(marked_image_path))
        assert(flask_image_path[:len(root_dir)]) == root_dir
        flask_image_path = flask_image_path[len(root_dir):].replace("\\", "/")
        out_text = '\n'.join(out_text)
        image_paths_and_texts.append((flask_image_path, out_text,))
        file_names.append((marked_image_path, recognized_text_path))
    form = ResultsForm(results_list=json.dumps(file_names))
    return render_template(template, form=form, image_paths_and_texts=image_paths_and_texts)

@app.route("/email", methods=['GET', 'POST'])
@login_required
@mobile_template('{m/}email.html')
def email(template):
    class Form(FlaskForm):
        e_mail = StringField('E-mail', validators=[DataRequired()])
        title = StringField('Заголовок письма')
        as_attachment = BooleanField("отправить как вложение")
        submit = SubmitField('Отправить')
    form = Form()
    if form.validate_on_submit():
        results_list = json.loads(request.values['results_list'])
        title = form.title.data or "Распознанный Брайль: " + Path(results_list[0][0]).with_suffix('').with_suffix('').name
        send_mail(form.e_mail.data, results_list, title)
        return redirect(url_for('index',
                                has_public_confirm=request.values['has_public_confirm'],
                                lang=request.values['lang'],
                                find_orientation=request.values['find_orientation'],
                                process_2_sides=request.values['process_2_sides']))
    form = Form(e_mail=current_user.e_mail)
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


def send_mail(to_address, results_list, subject):
    """
    Sends results to e-mail as text(s) + image(s)
    :param to_address: destination email as str
    :param results_list: list of tuples with file names(txt or jpg)
    :param subject: message subject or None
    """
    # create message object instance
    msg = MIMEMultipart()
    msg['From'] = "AngelinaReader <{}>".format(Config.SMTP_FROM)
    msg['To'] = to_address
    msg['Subject'] = subject if subject else "Распознанный Брайль"
    # attach image to message body
    for file_names in results_list:
        for file_name in reversed(file_names):  # txt before jpg
            if Path(file_name).suffix == ".txt":
                txt = Path(file_name).read_text(encoding="utf-8")
                attachment = MIMEText(txt, _charset="utf-8")
                attachment.add_header('Content-Disposition', 'inline', filename=Path(file_name).name)
            elif Path(file_name).suffix == ".jpg":
                attachment = MIMEImage(Path(file_name).read_bytes())
                attachment.add_header('Content-Disposition', 'inline', filename=Path(file_name).name)
            else:
                assert False, str(file_name)
            msg.attach(attachment)

    # create server and send
    server = smtplib.SMTP("{}: {}".format(Config.SMTP_SERVER, Config.SMTP_PORT))
    server.starttls()
    server.login(Config.SMTP_FROM, Config.SMTP_PWD)
    server.sendmail(msg['From'], msg['To'], msg.as_string())
    server.quit()


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
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        app.run(host='0.0.0.0', threaded=True)


if __name__ == "__main__":
    run()