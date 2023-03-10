import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_demo.settings")
app = Celery("web_demo")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
