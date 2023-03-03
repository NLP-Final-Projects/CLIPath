from django.http import HttpResponseBadRequest
from django.shortcuts import render, get_object_or_404, redirect

from .forms import CreateTaskForem
from .models import Task
from .tasks import schedule_task

# Create your views here.
def main_page(request):
    if request.GET.get('task_id'):
        return redirect('task_result', request.GET.get('task_id'))
    else:
        return render(request, template_name='main.html')
def create_task(request):
    if request.method == "GET":
        return render(request, template_name='create_task.html')
    elif request.method == "POST":
        form = CreateTaskForem(data=request.POST, files=request.FILES)
        if form.is_valid():
            task = form.save()
            task.status = Task.SCHEDULED
            task.save()
            schedule_task.apply_async(args=[str(task.task_id)])
            return render(request, context={'create': True, 'task': task}, template_name='task_result.html')
        else:
            return render(request, context={'errors': form.errors}, template_name='create_task.html')
    else:
        return HttpResponseBadRequest()


def task_result(request, uuid):
    task = get_object_or_404(Task, task_id=uuid)
    return render(request, context={'task': task}, template_name='task_result.html')


