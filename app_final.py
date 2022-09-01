import os
from fastapi import FastAPI
from schema import Response
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger
import torch
import torch.nn as nn
import hashlib
from catboost import CatBoostClassifier
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
POSTGRES_DATABASE = os.environ.get("POSTGRES_DATABASE")


def get_model_path_control(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        model_path = '/workdir/user_input/model_control'
    else:
        model_path = path
    return model_path


def get_model_path_test(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        model_path = '/workdir/user_input/model_test'
    else:
        model_path = path
    return model_path


def load_data_sql(query: str):
    engine = create_engine(
        "postgresql://" + POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@"
        + str(POSTGRES_HOST) + ":" + POSTGRES_PORT + "/" + POSTGRES_DATABASE)
    conn = engine.connect().execution_options(stream_results=True)
    parts = []

    for part in pd.read_sql(query, conn, chunksize=150000):
        parts.append(part)
    conn.close()
    return pd.concat(parts, ignore_index=True)


def true_pos_col(df):
    true_columns_list = ['hour', 'month', 'day', 'Vec_text_0',
                         'Vec_text_1', 'Vec_text_2', 'Vec_text_3', 'Vec_text_4',
                         'Vec_text_5', 'Vec_text_6', 'Vec_text_7', 'Vec_text_8',
                         'Vec_text_9', 'Vec_text_10', 'Vec_text_11', 'Vec_text_12',
                         'Vec_text_13', 'Vec_text_14', 'Vec_text_15', 'Vec_text_16',
                         'Vec_text_17', 'Vec_text_18', 'Vec_text_19', 'Vec_text_20',
                         'Vec_text_21', 'Vec_text_22', 'Vec_text_23', 'Vec_text_24',
                         'Vec_text_25', 'Vec_text_26', 'Vec_text_27', 'Vec_text_28',
                         'Vec_text_29', 'Vec_text_30', 'Vec_text_31', 'Vec_text_32',
                         'Vec_text_33', 'Vec_text_34', 'Vec_text_35', 'Vec_text_36',
                         'Vec_text_37', 'Vec_text_38', 'Vec_text_39', 'Vec_text_40',
                         'Vec_text_41', 'Vec_text_42', 'Vec_text_43', 'Vec_text_44',
                         'Vec_text_45', 'Vec_text_46', 'Vec_text_47', 'Vec_text_48',
                         'Vec_text_49', 'Vec_text_50', 'Vec_text_51', 'Vec_text_52',
                         'Vec_text_53', 'Vec_text_54', 'Vec_text_55', 'Vec_text_56',
                         'Vec_text_57', 'Vec_text_58', 'Vec_text_59', 'Vec_text_60',
                         'Vec_text_61', 'Vec_text_62', 'Vec_text_63', 'Vec_text_64',
                         'Vec_text_65', 'Vec_text_66', 'Vec_text_67', 'Vec_text_68',
                         'Vec_text_69', 'Vec_text_70', 'Vec_text_71', 'Vec_text_72',
                         'Vec_text_73', 'Vec_text_74', 'Vec_text_75', 'Vec_text_76',
                         'Vec_text_77', 'Vec_text_78', 'Vec_text_79', 'count',
                         'topic_covid', 'topic_entertainment', 'topic_movie',
                         'topic_politics', 'topic_sport', 'topic_tech', 'age', 'gender_1',
                         'code_country', 'code_city', 'exp_group_1', 'exp_group_2',
                         'exp_group_3', 'exp_group_4', 'os_iOS', 'source_organic',
                         'u_vl_movie', 'u_vl_covid', 'u_vl_sport', 'u_vl_entertainment',
                         'u_vl_politics', 'u_vl_business', 'u_vl_tech']
    return df[true_columns_list]


def load_features():
    """ posts = pd.read_csv('https://drive.google.com/uc?id=113Nw01J9oMm7klMs3cGEjbxDY8LMXs0n'),
        posts_features = pd.read_csv('https://drive.google.com/uc?id=1dl5cFOrXh1C6MlNSAhAoj5sOnlVRkIvF'),
        user_features_control = pd.read_csv('https://drive.google.com/uc?id=1FbLU1QEB-MOG-j6XMyktCUmYJC10V4SF'),
        user_features_test = pd.read_csv('https://drive.google.com/uc?id=1qvpO9z-kPrYMCqbt-wdsDJcs0BAru9A8'),
        liked_posts = pd.read_csv('https://drive.google.com/uc?id=1TR9xQ9xPmu5_9DdGoVnYk5HqjxcazEPo') """

    # Датасет по постам
    logger.info('sql loading post data')
    posts = pd.read_sql(
        """SELECT * FROM public.post_text_df""",

        con="postgresql://" + POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@"
            + str(POSTGRES_HOST) + ":" + POSTGRES_PORT + "/" + POSTGRES_DATABASE)

    # Фичи по постам
    logger.info('sql loading post features')
    posts_features = pd.read_sql("""SELECT * FROM public.posts_features""",
                                 con="postgresql://" + POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@"
                                     + str(POSTGRES_HOST) + ":" + POSTGRES_PORT + "/" + POSTGRES_DATABASE)

    # Фичи по юзерам
    logger.info('sql loading user data')
    user_features_control = pd.read_sql(
        """SELECT * FROM public.user_data""",

        con="postgresql://" + POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@"
            + str(POSTGRES_HOST) + ":" + POSTGRES_PORT + "/" + POSTGRES_DATABASE)

    user_features_test = pd.read_sql(
        """SELECT * FROM public.user_features""",
        con="postgresql://" + POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@"
            + str(POSTGRES_HOST) + ":" + POSTGRES_PORT + "/" + POSTGRES_DATABASE)

    # Пролайканные посты
    logger.info('sql loading liked posts')
    like_post_q = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action='like'"""
    liked_posts = load_data_sql(like_post_q)

    return [liked_posts, user_features_test, posts, posts_features, user_features_control]


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(107, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.layer_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.layer_3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.layer_out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_out(x)
        return x


def load_model_control():
    model_path = get_model_path_control("model_control")
    cb_model = CatBoostClassifier()
    cb_model.load_model(model_path)
    return cb_model


def load_model_test():
    model_path = get_model_path_test("model_test")
    fc_model = FC()
    fc_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return fc_model


def get_exp_group(user_id: int) -> str:  # 2 groups = (50% / 50%)
    salt = 'ლ(ಠ_ಠ ლ)'
    user = str(user_id)
    value_str = user + salt
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    if value_num % 100 < 50:
        return 'control'
    else:
        return 'test'


def get_rec_feed(id: int, time: datetime, limit: int, exp_group, model_use):
    # Берем модель
    model = model_use

    # Фичи по юзерам
    logger.info('create user features')

    if exp_group == 'control':
        user_features = features[4]
        user_features = user_features.loc[user_features.user_id == id]
        user_features = user_features.drop(['user_id', 'source', 'os', 'country'], axis=1)
    else:
        user_features = features[1]
        user_features = user_features.loc[user_features.user_id == id]
        user_features = user_features.drop(['user_id'], axis=1)

    # Фичи по постам
    logger.info('create post features')
    if exp_group == 'control':
        post_features = features[2].drop('text', axis=1)
    else:
        post_features = features[3]
    content = features[2][['post_id', 'text', 'topic']]

    # Объеденение фичей
    logger.info('user + post features')
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_post_features = post_features.assign(**add_user_features)
    user_post_features = user_post_features.set_index('post_id')

    # Добавление фич по времени
    logger.info('add time info')
    user_post_features['hour'] = time.hour
    user_post_features['month'] = time.month
    user_post_features['day'] = time.day

    # Формирование предсказаний
    logger.info('predict')
    if exp_group == 'control':
        predicts = model.predict_proba(user_post_features)[:, 1]
        user_post_features['predicts'] = predicts
    else:
        model.eval()
        with torch.no_grad():
            predicts = model(torch.tensor(true_pos_col(user_post_features).values, dtype=torch.float32))
            user_post_features['predicts'] = torch.sigmoid(predicts)

    # Чистка от пролайканных постов
    logger.info('filter from liked posts')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered = user_post_features[~user_post_features.index.isin(liked_posts)]

    # Рекомендация
    logger.info('give recommendations')
    rec_posts = filtered.sort_values('predicts')[-limit:].index
    recommendations = [{
        'id': i,
        'text': content[content.post_id == i].text.values[0],
        'topic': content[content.post_id == i].topic.values[0]}
        for i in rec_posts]
    return Response(**{'exp_group': exp_group,
                       'recommendations': recommendations})


logger.info('---Start_app---')
logger.info('loading features')
features = load_features()
logger.info('loading models')
model_control = load_model_control()
model_test = load_model_test()


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    logger.info('---Build_recommendations---')
    exp_group = get_exp_group(id)
    if exp_group == 'control':
        logger.info('use control model')
        model_use = model_control
    else:
        logger.info('use test model')
        model_use = model_test
    return get_rec_feed(id, time, limit, exp_group, model_use)
