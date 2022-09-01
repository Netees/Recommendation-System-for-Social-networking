import app_final as application
from fastapi.testclient import TestClient
from datetime import datetime

client = TestClient(application.app)
user_id = 201      # 200 -> control, 201 -> test
time = datetime(2021, 12, 15, 20)

try:
    r = client.get(f"/post/recommendations/",
                   params={'id': user_id, 'time': time, 'limit': 5},
                   )
except Exception as e:
    raise ValueError(f'X ошибка при выполнении запроса')

print(r.json())
