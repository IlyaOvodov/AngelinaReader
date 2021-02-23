#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Описание интерфейсов между UI и алгоритмическим модулями
"""
from datetime import datetime
import os
from pathlib import Path
import sqlite3
import time
import timeit
import uuid

import local_config
import model.infer_retinanet as infer_retinanet

model_weights = 'model.t7'

recognizer = None

class User:
    """
    Пользователь системы и его атрибуты
    Экземпляры класса создавать через AngelinaSolver.find_user или AngelinaSolver.register_user
    """
    def __init__(self, id, user_dict):
        """
        Ниже список атрибутов для demo
        Все атрибуты - read only, изменять через вызовы соответствующих методов
        """
        self.id = id  # уникальный для системы id пользователя. Присваивается при регистрации.
        self.name = user_dict["name"]
        self.email = user_dict["email"]

        # Данные, с которыми юзер был найден через find_user или создан через register_user.
        # У пользователя может быть несколько способов входа, поэтому 
        self.network_name = user_dict.get("network_name")       # TODO понять как кодировать соцсети. Для регистрации через email = None
        self.network_id = user_dict.get("network_id")
        self.password_hash = user_dict.get("password_hash")

        # поля для Flask:
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
        
        
    def get_id(self):
        return self.id        
        
    def check_password(self, password_hash):
        """
        Проверяет пароль. Вызывать про логине по email.
        """
        return self.password_hash is None or self.password_hash == password_hash

    def set_name(self, name):
        """
        изменение имени ранее зарегистрированного юзера
        """
        raise NotImplementedError
        pass
        
    def set_email(self, email):
        """
        изменение email ранее зарегистрированного юзера
        """
        raise NotImplementedError
        pass
        
    def set_password(self, password):
        """
        Обновляет пароль. Вызывать про логине по email.
        """
        # TODO raise NotImplementedError
        pass
        
    def get_param_default(self, param_name):
        """
        Возвращает занчение по умолчанию для параметра param_name (установки по умолчанию параметров в разных формах)
        """
        raise NotImplementedError
        return 123
        
    def set_param_default(self, param_name, param_value):
        """
        Возвращает заначение по умолчанию для параметра param_name
        """
        raise NotImplementedError
        pass

def exec_sqlite(con, query, params, timeout=10):
    """
    Пытается выполнить команду над sqlite, при ошибке повторяет в течение timeout секунд.
    :param con: connection
    :param query: sql text
    :param params: tuple or dict of params
    :param timeout: seconds
    :return: result of query
    """
    t0 = timeit.default_timer()
    i = 0
    while True:
        i += 1
        try:
            res = con.cursor().execute(query, params).fetchall()
            con.commit()
            return res
        except sqlite3.OperationalError as e:
            t = timeit.default_timer()
            if t > t0 + timeout:
                raise Exception("{} {} times {} to {} for {}".format(str(e), i, t, t0, query))
            time.sleep(0.1)

class UserManager:
    def __init__(self, db_file_name):
        self.db_file_name = db_file_name

    def register_user(self, name, email, password_hash, network_name, network_id):
        """
        Регистрирует юзера с заданными параметрами через email-пароль или через соцсеть.
        name, email указываются всегда.
        указывается или password (при регистрации через email) или network_name + network_id (при регистрации через сети)
        Проверяет, что юзера с таким email с регистрацией по email (при регистрации через email)
        или network_name + network_id (при регистрации через сети), не существует.
        Если существует, выдает exception.
        Возвращает класс User
        """
        id = uuid.uuid4().hex
        new_user = {
            "id": id,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "network_name": network_name,
            "network_id": network_id,
            "reg_date": datetime.now()
        }
        existing_user = self.find_user(network_name=network_name, network_id=network_id, email=email)
        assert not existing_user, ("such user already exists", network_name, network_id, email)
        con = self._sql_conn()
        exec_sqlite(con, "insert into users(id, name, email, network_name, network_id, password_hash, reg_date) values(:id, :name, :email, :network_name, :network_id, :password_hash, :reg_date)", new_user)
        return User(id, new_user)

    def find_user(self, network_name=None, network_id=None, email=None, id=None):
        """
        Возвращает объект User по регистрационным данным: id или паре network_name+network_id или регистрации по email (для этого указать network_name = None или network_name = "")
        Если юзер не найден, возвращает None
        """
        con = self._sql_conn()
        con.row_factory = sqlite3.Row
        if id:
            assert not network_name and not network_id and not email, ("incorrect call to find_user 1", network_name, network_id, email)
            query = ("select * from users where id = ?", (id,))
        elif network_name or network_id:
            assert network_name and network_id, ("incorrect call to find_user 2", network_name, network_id, email)
            query = ("select * from users where network_name = ? and network_id = ?", (network_name,network_id,))
        else:
            assert email and not network_name and not network_id, ("incorrect call to find_user 3", network_name, network_id, email)
            query = ("select * from users where email = ? and (network_name is NULL or network_name='') and (network_id is NULL or network_id='')", (email,))
        res = exec_sqlite(con, query[0], query[1])
        if len(res):
            user_dict = dict(res[0])  # sqlite row -> dict
            assert len(res) <= 1, ("more then 1 user found", user_dict)
            user = User(id=user_dict["id"], user_dict=user_dict)
            return user
        return None  # Nothing found


    def find_users_by_email(self, email):
        """
        Используется для проверки, что юзер случайно не регистрируется повторно
        Возвращает Dict(Dict) пользователей с указанным е-мейлом: id: user_dict.
        Может вернуть пустой словарь, словарь из одного или список из нескольких юзеров.
        """
        con = self._sql_conn()
        con.row_factory = sqlite3.Row
        res = exec_sqlite(con, "select * from users where email = ?", (email,))
        found = dict()
        for row in res:
            user_dict = dict(row)  # sqlite row -> dict
            found[user_dict["id"]] = user_dict
        return found

    def _sql_conn(self):
        timeout = 0.1
        new_db = not os.path.isfile(self.db_file_name)
        con = sqlite3.connect(str(self.db_file_name), timeout=timeout)
        if new_db:
            con.cursor().execute(
                "CREATE TABLE users(id text PRIMARY KEY, name text, email text, network_name text, network_id text, password_hash text, reg_date text)")
            self._convert_from_json(con)
            con.commit()
        return con

    def _convert_from_json(self, con):
        import json
        json_file = os.path.splitext(self.db_file_name)[0]+'.json'
        if os.path.isfile(json_file):
            with open(json_file, encoding='utf-8') as f:
                all_users = json.load(f)
            for id, user_dict in all_users.items():
                con.cursor().execute("INSERT INTO users(id, name, email) VALUES(?, ?, ?)",
                                     (id, user_dict["name"], user_dict["email"]))


class AngelinaSolver:
    """
    Обеспечивает интерфейс с вычислительной системой: пользователи, задачи и результаты обработки
    """
    def __init__(self, data_root_path="static/data"):
        self.data_root = Path(data_root_path)
        self.results_root = self.data_root / 'results'
        self.tasks_root = self.data_root / 'tasks'

        os.makedirs(self.data_root, exist_ok=True)
        self.user_manager = UserManager(self.data_root / "all_users.db")

        global recognizer
        if recognizer is None:
            print("infer_retinanet.BrailleInference()")
            t = timeit.default_timer()
            recognizer = infer_retinanet.BrailleInference(
                params_fn=os.path.join(local_config.data_path, 'weights', 'param.txt'),
                model_weights_fn=os.path.join(local_config.data_path, 'weights', model_weights),
                create_script=None)
            print(timeit.default_timer() - t)
        self.recognizer = recognizer

    help_articles = ["test_about", "test_photo"]
    help_contents = {
        "rus": {
            "test_about": {"title": "О программе",
                           "announce": "Это очень крутая программа! <b>Не пожалеете</b>! Просто нажмите кнопку",
                           "text": "Ну что вам еще надо. <b>Вы не верите</b>?"},
            "test_photo": {"title": "Как сделать фото",
                           "announce": "Чтобы сделать фото нужен фотоаппарат",
                           "text": "Просто нажмите кнопку!"}
        },
        "eng": {
            "test_about": {"title": "About",
                           "announce": "It a stunning program! <b>Dont miss it</b>! Just press the button",
                           "text": "Why don't you believe! What do you need <b>more</b>!"},
            "test_photo": {"title": "How to make photo",
                           "announce": "In order to make photo you need a camera",
                           "text": "You really need. Believe me!"}
        }
    }

    def help_list(self, target_language, search_qry):
        """
        Возвращаем список материалов для страницы help. Поскольку создавать html файл для каждой информационной статьи
        не самая лучшая идея, это лучше делать через вывод материалов из БД
        """
        total_list = [{ **{tag: self.help_contents[target_language][slug][tag] for tag in ["title", "announce"]}, **{"slug": slug} }
                for slug in self.help_articles]
        if search_qry:
            return total_list[:1]
        return  total_list


        # [
        #     {"title": "О программе", "announce": "Это очень крутая программа! Не пожалеете! Просто нажмите кнопку", "slug": "test_about"},
        #     {"title": "Как сделать фото", "announce": "Чтобы сделать фото нужен фотоаппарат", "slug": "test_photo"},
        # ]

    def help_item(self, target_language, slug):
        """
        Возвращаем материал для странии help.
        """
        return self.help_contents[target_language][slug]
        # {"title":"name","text":"def_text"}


    #Работа с записями пользователей: создание (регистрация), обработка логина:
    def register_user(self, name, email, password_hash, network_name, network_id):
        """
        Регистрирует юзера с заданными параметрами через email-пароль или через соцсеть.
        name, email указываются всегда. 
        указывается или password (при регистрации через email) или network_name + network_id (при регистрации через сети)
        Проверяет, что юзера с таким email с регистрацией по email (при регистрации через email)
        или network_name + network_id (при регистрации через сети), не существует.
        Если существует, выдает exception.
        Возвращает класс User
        """
        return self.user_manager.register_user(name=name, email=email, password_hash=password_hash, network_name=network_name, network_id=network_id)

    def find_user(self, network_name=None, network_id=None, email=None, id=None):
        """
        Возвращает объект User по регистрационным данным: паре network_name+network_id или регистрации по email (для этого указать network_name = None или network_name = "")
        Если юзер не найден, возвращает None
        """
        return self.user_manager.find_user(network_name, network_id, email, id)

    def find_users_by_email(self, email):
        """
        Используется для проверки, что юзер случайно не регистрируется повторно
        Возвращает Dict(Dict) пользователей с указанным е-мейлом: id: used_dict.
        Может вернуть пустой словарь, словарь из одного или список из нескольких юзеров.
        """
        return self.user_manager.find_users_by_email(email)
    
    #GVNC
    TMP_RESILTS = ['IMG_20210104_093412', 'IMG_20210104_093217']

    # собственно распознавание
    def process(self, user_id, img_paths, param_dict, timeout=0):
        """
        user: User ID or None для анонимного доступа
        img_paths: полный пусть к загруженному изображению, pdf или zip или список (list) полных путей к изображению
        param_dict: включает
            lang: выбранный пользователем язык ('RU', 'EN')
            find_orientation: bool, поиск ориентации
            process_2_sides: bool, распознавание обеих сторон
            has_public_confirm: bool, пользователь подтвердило публичную доступность результатов
        timeout: время ожидания результата. Если None или < 0 - жадть пока не будет завершена. 0 поставить в очередь и не ждать.
        
        Ставит задачу в очередь на распознавание и ждет ее завершения в пределах timeout.
        После успешной загрузки возвращаем id материалов в системе распознавания или False если в процессе обработки 
        запроса возникла ошибка. Далее по данному id мы переходим на страницу просмотра результатов данного распознавнаия
        """
        task_id = Path(img_paths).stem
        self.last_task_id = task_id

        #lang, find_orientation, process_2_sides, has_public_confirm

        ext = Path(img_paths).suffix[1:]  # exclude leading dot
        if ext == 'zip':
            results_list = self.recognizer.process_archive_and_save(img_paths, self.results_root,
                                                   lang=param_dict['lang'], extra_info=param_dict,
                                                   draw_refined=self.recognizer.DRAW_NONE,
                                                   remove_labeled_from_filename=False,
                                                   find_orientation=param_dict['find_orientation'],
                                                   align_results=True,
                                                   process_2_sides=param_dict['process_2_sides'],
                                                   repeat_on_aligned=False)

        else:
            results_list = self.recognizer.run_and_save(img_paths, self.results_root, target_stem=None,
                                                   lang=param_dict['lang'], extra_info=param_dict,
                                                   draw_refined=self.recognizer.DRAW_NONE,
                                                   remove_labeled_from_filename=False,
                                                   find_orientation=param_dict['find_orientation'],
                                                   align_results=True,
                                                   process_2_sides=param_dict['process_2_sides'],
                                                   repeat_on_aligned=False)
        if results_list is None:
            return False
        # full path -> relative to data path
        self.last_results_list = list()
        for marked_image_path, recognized_text_path, _ in results_list:
            marked_image_path = str(Path(marked_image_path).relative_to(self.data_root))
            recognized_text_path =  str(Path(recognized_text_path).relative_to(self.data_root))
            recognized_braille_path = str(Path(recognized_text_path).with_suffix(".brl"))  # TODO GVNC
            self.last_results_list.append((marked_image_path, recognized_text_path, recognized_braille_path))
        return task_id
        
    def is_completed(self, task_id):
        """
        Проверяет, завершена ли задача с заданным id
        """
        """
        В тестовом варианте отображется как не готовый в течение 2 с после загрузки
        """
        return True  # TODO GVNC

    def get_results(self, task_id):
        """
        Возвращает результаты распознавания по задаче task_id.
        Не проверяет, что задача была поставлена этим пользователем. Это ответственность вызывающей стороны.
        
        Возвращает словарь с полями:
            {"name": str,
             "create_date": datetime,
             "protocol": путь к protocol.txt
			 "item_data": список (list) результатов по количеству страниц в задании. 
			 Каждый элемени списка - tuple из полных путей к файлам с изображением, распознанным текстом, распознанным брайлем
            }
        """
        """
        В тестововм варианте по очереди выдается то 1 документ, то 2.
        """
        assert task_id == self.last_task_id

        return {"name":task_id,
                "create_date": datetime.strptime('2011-11-04 00:05:23', "%Y-%m-%d %H:%M:%S"), #"20200104 200001",
                "item_data": self.last_results_list,
                "protocol": task_id + ".protocol.txt"   # TODO
                }

    def get_tasks_list(self, user_id, count):
        """
        count - кол-во запиисей
        Возвращает список task_id задач для данного юзера, отсортированный от старых к новым
        """
        """
        В тестовом варианте возвращает 10 раз взятый список из 2 демо-результатов.
        При этом сначала все они показываются как не законченные. По мере моделирования расчетов показывается   
        более реалистично: пример выдается как не готовый 2 сек после запуска распознавания
        Публичный -приватный - через одного
        """
        if not user_id or user_id == "false":
            return []

        lst = [
                  {
                    "id":task,
                    "date": datetime.strptime('2011-11-04 00:05:23', "%Y-%m-%d %H:%M:%S"),  # "20200104 200001",  #datetime.fromisoformat('2011-11-04T00:05:23')
                    "name":task + ".jpg",
                    "img_url":"/static/data/results/pic.jpg",  # PIL.Image.Open("web_app/static/data/results/pic.jpg")
                    #"desc":"буря\nмглою\nнебо",
                    "desc":"I            B             101\nкоторые созвучны друг с дру-\r\nгом. например, в первых четырёх ~?~",
                    "public": i%2 ==0,
                    "sost": self.is_completed(task)
                   }
            for i, task in enumerate(AngelinaSolver.TMP_RESILTS)
        ]*10

        if count:
            lst = lst[:count]
        return lst



    CONTENT_IMAGE = 1
    CONTENT_TEXT = 2
    CONTENT_BRAILLE = 4
    CONTENT_ALL = CONTENT_IMAGE | CONTENT_TEXT | CONTENT_BRAILLE

    # отправка почты
    def send_results_to_mail(self, mail, item_id, parameters=None):
        """
        Отправляет результаты на to_email и/или разработчикам. Что-то одно - обязательно.
        results_list - список результатов такой же, как возвращает get_results(...)
        to_email - адрес или список адресов
        parameters - словарь с параметрами формирования и отправки письма, в т.ч.:
            title - заголовок
            comment - комментарий, ставится в начале письма
        """
        # raise NotImplementedError
        return True

    def get_user_emails(self, user_id):
        """
        Список адресов почты, куда пользователь отсылал письма
        :param user_id: string
        :return: list of e-mails
        """
        if not user_id or user_id == "false":
            return []
        return ["angelina-reader@ovdv.ru", "il@ovdv.ru", "iovodov@gmail.com"]


if __name__ == "__main__":
    core = AngelinaSolver()
    #r = core.register_user(name="керогаз Керогазов", email="b1@ovdv.ru", password_hash="", network_name="", network_id=None)
    r = core.register_user(name="керогаз Керогазов", email="b@ovdv.ru", password_hash="qqwe", network_name="nnn", network_id=123)
    print(r)
    r = core.find_user(id="0162c109d0614ec2bf8dd21d025e816b")
    print(r)
    r = core.find_user(email="b@ovdv.ru")
    print(r)
    r = core.find_users_by_email(email="b@ovdv.ru")
    print(r)
