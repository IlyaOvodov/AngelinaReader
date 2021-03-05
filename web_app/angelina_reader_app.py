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
from flask_uploads import UploadSet, configure_uploads
from flask_mobility import Mobility
from flask_mobility.decorators import mobile_template

import atexit
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import email.utils as email_utils
import smtplib
import time
import uuid
import os
import json
import signal
import sys
import argparse
from pathlib import Path
import socket

from .config import Config
from .angelina_reader_core import AngelinaSolver, VALID_EXTENTIONS

def startup_logger():
    hostname = socket.gethostname()
    def send_startup_email(what):
        """
        Sends results to e-mail as text(s) + image(s)
        :param to_address: destination email as str
        :param results_list: list of tuples with file names(txt or jpg)
        :param subject: message subject or None
        """
        # create message object instance
        txt = 'Angelina Reader is {} at {}'.format(what, hostname)
        msg = MIMEText(txt, _charset="utf-8")
        msg['From'] = "AngelinaReader <{}>".format(Config.SMTP_FROM)
        msg['To'] = 'Angelina Reader<angelina-reader@ovdv.ru>'
        msg['Subject'] = txt
        msg['Date'] = email_utils.formatdate()
        msg['Message-Id'] = email_utils.make_msgid(idstring=str(uuid.uuid4()), domain=Config.SMTP_FROM.split('@')[1])

        # create server and send
        server = smtplib.SMTP("{}: {}".format(Config.SMTP_SERVER, Config.SMTP_PORT))
        server.starttls()
        server.login(Config.SMTP_FROM, Config.SMTP_PWD)
        recepients = [msg['To']]
        server.sendmail(msg['From'], recepients, msg.as_string())
        server.quit()

    send_startup_email('started')

    atexit.register(send_startup_email, 'stopped')
    def signal_handler(sig, frame):
        send_startup_email('interrupted by caught {}'.format(sig))
        sys.exit(0)
    for s in set(signal.Signals):
        try:
            signal.signal(s, signal_handler)
        except:
            print('failed to set handler for signal {}'.format(s))

app = Flask(__name__)
Mobility(app)
app.config.from_object(Config)
login_manager = LoginManager(app)

data_root_path = Path(app.root_path) / app.config['DATA_ROOT']

core = AngelinaSolver(data_root_path=data_root_path)

@login_manager.user_loader
def load_user(user_id):
    return core.find_user(id=user_id)


@app.route("/", methods=['GET', 'POST'])
@app.route("/index", methods=['GET', 'POST'])
@mobile_template('{m/}index.html')
def index(template, is_mobile=False):
    class MainForm(FlaskForm):
        camera_file = FileField()
        file = FileField()
        agree = BooleanField("Я согласен")
        disgree = BooleanField("Возражаю")
        lang = SelectField("Язык текста", choices=[('RU', 'Русский'), ('EN', 'English'), ('LV', 'Latviešu'), ('UZ', 'Ўзбек'), ('GR', 'Ελληνικά')])
        find_orientation = BooleanField("Авто-ориентация")
        process_2_sides = BooleanField("Обе стороны")
    form = MainForm(agree=request.values.get('has_public_confirm', '') == 'True',
                    disgree=request.values.get('has_public_confirm', '') == 'False',
                    lang=request.values.get('lang', 'RU'),
                    find_orientation=request.values.get('find_orientation', 'True') == 'True',
                    process_2_sides=request.values.get('process_2_sides', 'False') == 'True'
                    )
    if form.validate_on_submit():
        file_data = form.camera_file.data or form.file.data
        if not file_data:
            flash('Необходимо загрузить файл')
            return render_template(template, form=form)
        if form.agree.data and form.disgree.data or not form.agree.data and not form.disgree.data:
            flash('Выберите один из двух вариантов (согласен/возражаю)')
            return render_template(template, form=form)
        filename = file_data.filename
        file_ext = Path(filename).suffix[1:].lower()
        if file_ext not in VALID_EXTENTIONS:
            flash('Не подхожящий тип файла {}: {}'.format(file_ext, filename))
            return render_template(template, form=form)

        user_id = current_user.get_id()
        extra_info = {'user': user_id,
                      'has_public_confirm': form.agree.data,
                      'lang': form.lang.data,
                      'find_orientation': form.find_orientation.data,
                      'process_2_sides': form.process_2_sides.data,
                      }
        task_id = core.process(user_id=user_id, file_storage=file_data, param_dict=extra_info)

        if not form.agree.data:
            return redirect(url_for('confirm',
                                    task_id=task_id,
                                    lang=form.lang.data,
                                    find_orientation=form.find_orientation.data,
                                    process_2_sides=form.process_2_sides.data))
        return redirect(url_for('results',
                                task_id=task_id,
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
                                task_id=request.values['task_id'],
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
    results_list = None
    task_id = request.values['task_id']
    if task_id:
        assert core.is_completed(task_id)
        results_list = core.get_results(task_id)
    if results_list is None:
        flash(
            'Ошибка обработки файла. Возможно, файл имеет неверный формат. Если вы считаете, что это ошибка, пришлите файл по адресу, указанному в низу страцины')
        return redirect(url_for('index',
                                has_public_confirm=request.values['has_public_confirm'],
                                lang=request.values['lang'],
                                find_orientation=request.values['find_orientation'],
                                process_2_sides=request.values['process_2_sides']))
    # convert OS path to flask html path
    image_paths_and_texts = list()
    file_names = list()
    for marked_image_path, recognized_text_path, recognized_braille_path in results_list["item_data"]:
        # полный путь к картинке -> "/static/..."
        # даннные для отображения в форме
        with open(data_root_path / recognized_text_path, encoding="utf-8") as f:
            out_text = ''.join(f.readlines())
        image_paths_and_texts.append(("/" + app.config['DATA_ROOT'] + "/" + marked_image_path, out_text, out_text,))

        # список c полными путями для передачи в форму почты
        file_names.append((str(data_root_path / marked_image_path), str(data_root_path / recognized_text_path)))  # список для

    form = ResultsForm(results_list=json.dumps(file_names))
    return render_template(template, form=form, image_paths_and_texts=image_paths_and_texts)

@app.route("/email", methods=['GET', 'POST'])
@login_required
@mobile_template('{m/}email.html')
def email(template):
    class Form(FlaskForm):
        e_mail = StringField('E-mail', validators=[DataRequired()])
        to_developers = BooleanField('Отправить разработчикам')
        title = StringField('Заголовок письма')
        comment = TextAreaField('Комментарий')
        as_attachment = BooleanField("отправить как вложение")
        submit = SubmitField('Отправить')
    form = Form()
    if form.validate_on_submit():
        results_list = json.loads(request.values['results_list'])
        title = form.title.data or "Распознанный Брайль: " + Path(results_list[0][0]).with_suffix('').with_suffix('').name.lower()
        send_mail(form.e_mail.data, results_list, title, form.to_developers.data, form.comment.data)
        return redirect(url_for('index',
                                has_public_confirm=request.values['has_public_confirm'],
                                lang=request.values['lang'],
                                find_orientation=request.values['find_orientation'],
                                process_2_sides=request.values['process_2_sides']))
    form = Form(e_mail=current_user.email)
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
        user = core.find_user(email=form.e_mail.data)  # TODO by network
        if user is None:
            flash('Пользователь не найден. Если вы - новый пользователь, зарегистрируйтесь')
            return redirect(url_for('register'))
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
        found_users = core.find_users_by_email(email=form.e_mail.data)
        if len(found_users):
            flash('Пользователь с таким E-mail уже существует')
            return redirect(url_for('register'))
        user = core.register_user(name=form.username.data, email=form.e_mail.data, password_hash=None, network_name=None, network_id=None)  # TODO only email registration now
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))
    return render_template(template, title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route("/donate", methods=['GET', 'POST'])
@mobile_template('{m/}donate.html')
def donate(template):
    return render_template(template)

def send_mail(to_address, results_list, subject, to_developers, comment):
    """
    Sends results to e-mail as text(s) + image(s)
    :param to_address: destination email as str
    :param results_list: list of tuples with file names(txt or jpg)
    :param subject: message subject or None
    """
    # create message object instance
    msg = MIMEMultipart()
    msg['From'] = "AngelinaReader <{}>".format(Config.SMTP_FROM)
    if to_developers:
        msg['To'] = to_address + ',Angelina Reader<angelina-reader@ovdv.ru>'
    else:
        msg['To'] = to_address
    msg['Subject'] = subject if subject else "Распознанный Брайль"
    msg['Date'] = email_utils.formatdate()
    msg['Message-Id'] = email_utils.make_msgid(idstring=str(uuid.uuid4()), domain=Config.SMTP_FROM.split('@')[1])
    attachment = MIMEText(comment + "\nLetter from: {}<{}>".format(current_user.name, current_user.email), _charset="utf-8")
    msg.attach(attachment)
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
    recepients = msg['To'].split(',')
    server.sendmail(msg['From'], recepients, msg.as_string())
    server.quit()


def run():
    parser = argparse.ArgumentParser(description='Angelina Braille reader web app.')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='enable debug mode (default: off)')
    args = parser.parse_args()
    debug = args.debug
    if not debug:
        startup_logger()
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