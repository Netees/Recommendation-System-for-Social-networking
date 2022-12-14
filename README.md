# Recommendation-System-for-Social-networking

## **Выпускной проект курса** [Start ML KARPOV.COURSES](https://karpov.courses/ml-start) (1 поток)

## Задача
Необходимо реализовать сервис, который будет для каждого юзера в любой момент времени возвращать посты, которые пользователю покажут в его ленте соцсети.

## Решение
В качестве метрики оценки постов для рекомендации используется ROC-AUC на таргете "поставил ли пользователь лайк посту".

1) Подготовка фич и обучение на них Catboost модели ([Control model](https://github.com/Netees/Recommendation-System-for-Social-networking/blob/main/Prep_features_and_learning_models/Model_control_Catboost.ipynb)) 
2) Подготовка фич и обучение на них Fully connected NN ([Test model](https://github.com/Netees/Recommendation-System-for-Social-networking/blob/main/Prep_features_and_learning_models/Model_test_FC_NN.ipynb))
3) Написание [приложения](https://github.com/Netees/Recommendation-System-for-Social-networking/blob/main/app_final.py), которое разбивает пользователей на группы(control & test) и для каждой из групп используется одна из двух реализованных моделей для выдачи рекомендации
4) [Проведение А/В тестирования](https://github.com/Netees/Recommendation-System-for-Social-networking/blob/main/AB%20testing.ipynb)

(Stack: sklearn, catboost, PyTorch, FastText, pandas, numpy, FastAPI, sqlalchemy и др.)

[**Тестирование приложения, какие рекомендации выдает**](https://github.com/Netees/Recommendation-System-for-Social-networking/blob/main/check_req.py)

## Описание данных
### Таблица user_data
Cодержит информацию о всех пользователях соц.сети

| Field name | Overview |
| :---: | --- |
| age |	Возраст пользователя (в профиле) |
| city |	Город пользователя (в профиле) |
| country |	Страна пользователя (в профиле) |
| exp_group |	Экспериментальная группа: некоторая зашифрованная категория |
| gender |	Пол пользователя |
| id |	Уникальный идентификатор пользователя |
| os |	Операционная система устройства, с которого происходит пользование соц.сетью |
| source |	Пришел ли пользователь в приложение с органического трафика или с рекламы |

### Таблица post_text_df
Содержит информацию о постах и уникальный ID каждой единицы с соответствующим ей текстом и топиком

| Field name | Overview |
| :---: | --- |
| id |	Уникальный идентификатор поста |
| text |	Текстовое содержание поста |
| topic |	Основная тематика |

### Таблица feed_data
Содержит историю о просмотренных постах для каждого юзера в изучаемый период. 

| Field name | Overview |
| :---: | --- |
| timestamp |	Время, когда был произведен просмотр |
| user_id |	id пользователя, который совершил просмотр |
| post_id |	id просмотренного поста |
| action |	Тип действия: просмотр или лайк |
| target |	1 у просмотров, если почти сразу после просмотра был совершен лайк, иначе 0. У действий like пропущенное значение. |

## Техническая спецификация

#### Endpoint GET /post/recommendations/

| Parameter |	Overview |
| :---: | --- |
| id |	ID user’а для которого запрашиваются посты |
| time |	Объект типа datetime: datetime.datetime(year=2021, month=1, day=3, hour=14) |
| limit |	Количество постов для юзера |
| exp_group | Группа, в которую попал пользователь |

#### Response

```
{
'exp_group': 'test',
'recommendations': [{
  "id": 345,
  "text": "COVID-19 runs wild....",
  "topic": "news"
}, 
{
  "id": 134,
  "text": "Chelsea FC wins UEFA..",
  "topic": "news"
}, 
...]}
```
