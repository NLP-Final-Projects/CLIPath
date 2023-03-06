from django import forms

from .models import Task, Query


class CreateTaskForem(forms.ModelForm):
    class Meta:
        model = Task
        fields = ['image', 'backend']
        widgets = {
            'backend': forms.RadioSelect
        }

    def is_valid(self):
        data = self.data
        for key in self.data.keys():
            if key.startswith('query-'):
                if not QueryForm(data={'text': self.data[key]}).is_valid():
                    self.add_error('image', f'text for query {key} not valid.')
                    return False
        return super(CreateTaskForem, self).is_valid()

    def save(self, commit=True):
        task = super(CreateTaskForem, self).save(commit)
        for key in self.data.keys():
            if key.startswith('query'):
                Query.objects.create(task=task, text=self.data[key])
        return task


class QueryForm(forms.ModelForm):
    class Meta:
        model = Query
        fields = ['text']
