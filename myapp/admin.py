
from django.contrib import admin
from . models import Data,Predicted,Ids

# Register your models here.
admin.site.register(Data)
admin.site.register(Predicted)
admin.site.register(Ids)