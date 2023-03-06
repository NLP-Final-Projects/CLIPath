from celery import shared_task

from .utils import baseline_pipe
from .models import Task

@shared_task()
def schedule_task(task_id):
    task = Task.objects.get(task_id=task_id)
    if task.backend == Task.BASELINE:
        baseline_pipe(task)
    else:
        pass
