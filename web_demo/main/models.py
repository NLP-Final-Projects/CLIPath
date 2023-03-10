import uuid

from django.db import models


# Create your models here.
class Task(models.Model):
    # todo: user
    CREATED = '0'
    SCHEDULED = '1'
    IN_PROCESS = '2'
    FINISHED = '3'
    FAILED = '4'
    STATUS_CHOICES = (
        (CREATED, 'Created'),
        (SCHEDULED, 'Scheduled in queue'),
        (IN_PROCESS, 'Processing'),
        (FINISHED, 'Finished'),
        (FAILED, 'Falied'),
    )
    BASELINE = '1'
    FINAL = '2'
    BACKEND_CHOICES = (
        (BASELINE, 'Baseline (CLIP)'),
        (FINAL, 'Final (CLIPath)')
    )
    status = models.CharField(max_length=1, choices=STATUS_CHOICES, default=CREATED)
    backend = models.CharField(max_length=1, default='1', choices=BACKEND_CHOICES)
    task_id = models.UUIDField(default=uuid.uuid4)
    create_date = models.DateField(auto_now_add=True)
    update_date = models.DateField(auto_now=True)
    image = models.FileField(upload_to='tasks/')

    def sorted_queries(self):
        if self.status != self.FINISHED:
            return self.queries
        return self.queries.all().order_by('-probability')

    def __str__(self):
        return str(self.task_id)


class Query(models.Model):
    task = models.ForeignKey(to=Task, on_delete=models.CASCADE, related_name='queries')
    probability = models.FloatField(blank=True, null=True)
    text = models.CharField(max_length=255)

    def __str__(self):
        return self.text
