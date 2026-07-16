from backend.core.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from backend.worker.celery_app import celery_app
from backend.worker.tasks import ping


def test_celery_uses_configured_redis_urls():
    assert celery_app.conf.broker_url == CELERY_BROKER_URL
    assert celery_app.conf.result_backend == CELERY_RESULT_BACKEND


def test_celery_serializes_tasks_and_results_as_json():
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]


def test_system_ping_task_returns_worker_status():
    assert ping.run() == {"status": "ok"}
