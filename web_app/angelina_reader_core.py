#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Описание интерфейсов между UI и алгоритмическим модулями
"""
import PIL
from datetime import datetime
import json
import os
import time
import uuid

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
        
    def check_password(self, password):
        """
        Проверяет пароль. Вызывать про логине по email.
        """
        raise NotImplementedError
        return True

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
        raise NotImplementedError
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
       

class UserManager:
    def __init__(self, user_file_name):
        self.user_file_name = user_file_name

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
            "name": name,
            "email": email,
        }
        if password_hash:
            new_user["password_hash"] = password_hash
        if network_name:
            new_user["network_name"] = network_name
        if network_id:
            new_user["network_id"] = network_id
        self._update_users_dict(id, new_user)
        return User(id, new_user)

    def find_user(self, id=None, network_name=None, network_id=None, email=None):
        """
        Возвращает объект User по регистрационным данным: id или паре network_name+network_id или регистрации по email (для этого указать network_name = None или network_name = "")
        Если юзер не найден, возвращает None
        """
        all_users = self._read_users_dict()
        if id:
            assert not network_name and not network_id and not email
            user_dict = all_users.get(id, None)
            if user_dict:
                user = User(id=id, user_dict=user_dict)
                return user
        else:
            if email:
                assert not network_name and not network_id
            found_user_dicts = dict()
            for id, user_dict in all_users.items():
                if (network_name and user_dict["network_name"] == network_name and network_id and user_dict["network_id"] == network_id
                    or email and user_dict["email"] == email):
                    found_user_dicts[id] = user_dict
            assert len(found_user_dicts) <= 1, found_user_dicts
            for id, user_dict in found_user_dicts.items():
                return User(id=id, user_dict=user_dict)
        return None  # Nothing found

    def find_users_by_email(self, email):
        """
        Используется для проверки, что юзер случайно не регистрируется повторно
        Возвращает Dict(Dict) пользователей с указанным е-мейлом: id: used_dict.
        Может вернуть пустой словарь, словарь из одного или список из нескольких юзеров.
        """
        all_users = self._read_users_dict()
        found = {id: user for id, user in all_users.items() if user['email'] == email}
        return found

    def _read_users_dict(self):
        if os.path.isfile(self.user_file_name):
            with open(self.user_file_name, encoding='utf-8') as f:
                all_users = json.load(f)
        else:
            all_users = dict()
        return all_users

    def _update_users_dict(self, id, user_dict):
        # TODO concurrent update
        all_users_dict = self._read_users_dict()
        all_users_dict[id] = user_dict
        with open(self.user_file_name, 'w', encoding='utf8') as f:
            json.dump(all_users_dict, f, sort_keys=True, indent=4, ensure_ascii=False)


class AngelinaSolver:
    """
    Обеспечивает интерфейс с вычислительной системой: пользователи, задачи и результаты обработки
    """
    def __init__(self, user_file_name):
        self.user_manager = UserManager(user_file_name)

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

    def find_user(self, id=None, network_name=None, network_id=None, email=None):
        """
        Возвращает объект User по регистрационным данным: паре network_name+network_id или регистрации по email (для этого указать network_name = None или network_name = "")
        Если юзер не найден, возвращает None
        """
        return self.user_manager.find_user(id, network_name, network_id, email)

    def find_users_by_email(self, email):
        """
        Используется для проверки, что юзер случайно не регистрируется повторно
        Возвращает Dict(Dict) пользователей с указанным е-мейлом: id: used_dict.
        Может вернуть пустой словарь, словарь из одного или список из нескольких юзеров.
        """
        return self.user_manager.find_users_by_email(email)
    
    #GVNC
    TMP_RESULT_SELECTOR = 1
    TMP_RESILTS = ['IMG_20210104_093412', 'IMG_20210104_093217']
    PREFIX = "" #"static/data/results/"

    # собственно распознавание
    def process(self, user, img_paths, lang, find_orientation, process_2_sides, has_public_confirm, timeout):
        """
        user: User class object or None для анонимного доступа
        img_paths: полный пусть к загруженному изображению, pdf или zip или список (list) полных путей к изображению
        lang: выбранный пользователем язык ('RU', 'EN')
        find_orientation: bool, поиск ориентации
        process_2_sides: bool, распознавание обеих сторон
        has_public_confirm: bool, пользователь подтвердило публичную доступность результатов
        timeout: время ожидания результата. Если None или < 0 - жадть пока не будет завершена. 0 поставить в очередь и не ждать.
        
        Ставит задачу в очередь на распознавание и ждет ее завершения в пределах timeout.
        Возвращает пару task_id (id задачи), завершена ли
        """
        if timeout and timeout > 0:
            time.sleep(timeout)

        AngelinaSolver.TMP_RESULT_SELECTOR = 1 - AngelinaSolver.TMP_RESULT_SELECTOR
        return AngelinaSolver.TMP_RESILTS[AngelinaSolver.TMP_RESULT_SELECTOR], True
        
    def is_completed(self, task_id):
        """
        Проверяет, завершена ли задача с заданным id
        """
        assert task_id in AngelinaSolver.TMP_RESILTS
        return True
        
    def get_results(self, task_id):
        """
        Возвращает результаты распознавания по задаче task_id.
        Не проверяет, что задача была поставлена этим пользователем. Это ответственность вызывающей стороны.
        
        Возвращает пару results_list, params.
        results_list - список (list) результатов по количеству страниц в задании. Каждый элемени списка - tuple из полных путей к файлам с изображением, распознанным текстом, распознанным брайлем
        params - полный путь к файлу с сохраненным словарем параметров распознавания
        """
        prefix = AngelinaSolver.PREFIX
        return [(prefix + task_id + ".marked.jpg", prefix + task_id + ".marked.txt", prefix + task_id + ".marked.brl",),], prefix + task_id + ".protocol.txt"
        
    def get_tasks_list(self, user):
        """
        Возвращает список task_id задач для данного юзера, отсортированный от старых к новым
        """
        return AngelinaSolver.TMP_RESILTS
        
    def get_task_breif(self, task_id):
        """
        Возвращает краткую информацию для заданнонй задачи для отображения в истории:
        время, имя, маленькую картинку, первые 3 строки текста, признак публичной доступности, признак, что задача завершена.
        TOFO открытый вопрос: 1) в каком формате время, 2) как возвращать картиннку (имя файла или битмап).
        """

        return (datetime.fromisoformat('2011-11-04T00:05:23'),
                task_id + ".jpg",
                PIL.Image.Open("web_app/static/data/results/pic.jpg"),
                "буря\nмглою\nнебо",
                True,
                True)

    CONTENT_IMAGE = 1
    CONTENT_TEXT = 2
    CONTENT_BRAILLE = 4
    CONTENT_ALL = CONTENT_IMAGE | CONTENT_TEXT | CONTENT_BRAILLE

    # отправка почты
    def send_results_to_mail(self, results_list, to_email, send_to_developers=False, title="", comment="", content_options=CONTENT_ALL):
        """
        Отправляет результаты на to_email и/или разработчикам. Что-то одно - обязательно.
        results_list - список результатов такой же, как возвращает get_results(...)
        to_email - адрес или список адресов
        title - заголовок
        comment - комментарий, ставится в начале письма
        content_options - опции, что выводить: картинку, тектс, текст на брайле или их комбинация (добавить enum  битовой маской)
        """
        raise NotImplementedError
        pass
    
    
    
        
        
    

        
    